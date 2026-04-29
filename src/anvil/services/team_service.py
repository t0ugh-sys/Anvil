from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..core.types import StopConfig
from ..task_graph import Task
from ..team_runtime import PersistentTeamRuntime, PersistentTeammateSpec


DeciderBuilder = Callable[[Any, Any], Any]
SkillsLoader = Callable[[Any], Any]


def parse_teammate(value: str) -> tuple[str, str]:
    raw = value.strip()
    if ':' not in raw:
        raise ValueError('teammate must be NAME:ROLE')
    name, role = raw.split(':', 1)
    name = name.strip()
    role = role.strip()
    if not name or not role:
        raise ValueError('teammate must be NAME:ROLE')
    return name, role


def parse_team_message(value: str) -> tuple[str, str]:
    raw = value.strip()
    if '=' not in raw:
        raise ValueError('message must be TARGET=BODY')
    target, body = raw.split('=', 1)
    target = target.strip()
    body = body.strip()
    if not target or not body:
        raise ValueError('message must be TARGET=BODY')
    return target, body


def spawn_team_runtime(
    args,
    *,
    decider_builder: DeciderBuilder,
    skills_loader: SkillsLoader,
) -> tuple[PersistentTeamRuntime, Path, list[PersistentTeammateSpec]]:
    workspace_root = Path(args.workspace).resolve()
    team_root = (workspace_root / args.team_dir).resolve()
    runtime = PersistentTeamRuntime(team_root)
    decider = decider_builder(args, skills_loader(args))
    teammate_specs: list[PersistentTeammateSpec] = []
    for item in args.teammate:
        name, role = parse_teammate(item)
        teammate_specs.append(
            PersistentTeammateSpec(
                name=name,
                role=role,
                workspace_root=workspace_root,
                decider=decider,
                stop=StopConfig(max_steps=args.max_steps, max_elapsed_s=args.timeout_s),
                skills=tuple(args.skills),
            )
        )
    for spec in teammate_specs:
        runtime.spawn_teammate(spec)
    return runtime, team_root, teammate_specs


def append_startup_tasks(runtime: PersistentTeamRuntime, task_bodies: list[str], sender: str) -> None:
    if not task_bodies:
        return
    runtime.replace_task_graph(
        [
            Task(id=f'task_{index}', title=body, goal=body)
            for index, body in enumerate(task_bodies, start=1)
        ]
    )
    runtime.dispatch_ready_tasks(sender=sender)


def run_team_run_command(
    args,
    *,
    decider_builder: DeciderBuilder,
    skills_loader: SkillsLoader,
) -> int:
    runtime, team_root, teammate_specs = spawn_team_runtime(
        args,
        decider_builder=decider_builder,
        skills_loader=skills_loader,
    )
    append_startup_tasks(runtime, args.task, args.sender)

    expected_replies = 0
    task_mode = bool(args.task)
    for item in args.message:
        recipient, body = parse_team_message(item)
        runtime.send_message(recipient, body, sender=args.sender)
        expected_replies += 1
    for body in args.broadcast:
        runtime.broadcast(body, sender=args.sender)
        expected_replies += len(teammate_specs)

    replies: list[dict[str, Any]] = []
    deadline = datetime.now(timezone.utc).timestamp() + args.service_timeout_s
    try:
        while datetime.now(timezone.utc).timestamp() < deadline:
            inbox = runtime.inbox_store.drain(args.sender)
            for item in inbox:
                replies.append(item.to_dict())
            tasks_still_active = task_mode and runtime.has_active_tasks()
            if expected_replies > 0 and len(replies) >= expected_replies and not tasks_still_active:
                break
            if expected_replies == 0 and not tasks_still_active:
                break
            time.sleep(args.poll_interval_s)
    finally:
        runtime.shutdown_all(sender=args.sender, timeout_s=min(5.0, args.service_timeout_s))

    payload = {
        'team_dir': str(team_root),
        'members': [member.to_dict() for member in runtime.config_store.load().members],
        'replies': replies,
        'tasks': runtime.load_task_graph().to_dict().get('tasks', []),
    }
    if args.output == 'json':
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(f'team_dir: {team_root}')
        print(f'members: {[member["name"] for member in payload["members"]]}')
        for reply in replies:
            print(f'{reply["sender"]} -> {reply["recipient"]}: {reply["body"]}')
    return 0


def run_team_serve_command(
    args,
    *,
    decider_builder: DeciderBuilder,
    skills_loader: SkillsLoader,
) -> int:
    runtime, team_root, teammate_specs = spawn_team_runtime(
        args,
        decider_builder=decider_builder,
        skills_loader=skills_loader,
    )
    append_startup_tasks(runtime, args.task, args.sender)
    for item in args.message:
        recipient, body = parse_team_message(item)
        runtime.send_message(recipient, body, sender=args.sender)
    for body in args.broadcast:
        runtime.broadcast(body, sender=args.sender)

    replies: list[dict[str, Any]] = []
    deadline = datetime.now(timezone.utc).timestamp() + args.service_timeout_s if args.service_timeout_s > 0 else None
    idle_since = datetime.now(timezone.utc).timestamp()
    try:
        while True:
            runtime.dispatch_ready_tasks(sender='scheduler')
            inbox = runtime.inbox_store.drain(args.sender)
            for item in inbox:
                replies.append(item.to_dict())

            active = runtime.has_active_tasks() or runtime.has_pending_member_messages()
            if active:
                idle_since = datetime.now(timezone.utc).timestamp()
            if args.idle_exit_s > 0 and not active and (datetime.now(timezone.utc).timestamp() - idle_since) >= args.idle_exit_s:
                break
            if deadline is not None and datetime.now(timezone.utc).timestamp() >= deadline:
                break
            if runtime.all_teammates_shutdown():
                break
            time.sleep(args.poll_interval_s)
    finally:
        if not runtime.all_teammates_shutdown():
            runtime.shutdown_all(sender=args.sender, timeout_s=min(5.0, args.timeout_s))

    payload = {
        'team_dir': str(team_root),
        'mode': 'service',
        'members': [member.to_dict() for member in runtime.config_store.load().members],
        'replies': replies,
        'tasks': runtime.load_task_graph().to_dict().get('tasks', []),
        'teammates': [spec.name for spec in teammate_specs],
    }
    if args.output == 'json':
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(f'team_dir: {team_root}')
        print('mode: service')
        print(f'teammates: {[spec.name for spec in teammate_specs]}')
        print(f'replies: {len(replies)}')
    return 0


def run_team_add_task_command(args) -> int:
    workspace_root = Path(args.workspace).resolve()
    runtime = PersistentTeamRuntime((workspace_root / args.team_dir).resolve())
    metadata: dict[str, Any] = {}
    if args.role:
        metadata['role'] = args.role
    task = Task(
        id=args.task_id or f'task_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}',
        title=args.title or args.goal,
        goal=args.goal,
        dependencies=tuple(args.depends_on),
        assignee=args.assignee or None,
        metadata=metadata,
    )
    graph = runtime.add_task(task)
    runtime.dispatch_ready_tasks(sender=args.sender)
    payload = {
        'team_dir': str((workspace_root / args.team_dir).resolve()),
        'task': graph.get_task(task.id).to_dict(),
    }
    if args.output == 'json':
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(f'added task: {task.id}')
    return 0


def run_team_send_command(args) -> int:
    workspace_root = Path(args.workspace).resolve()
    runtime = PersistentTeamRuntime((workspace_root / args.team_dir).resolve())
    runtime.send_message(args.to, args.message, sender=args.sender)
    print('ok')
    return 0


def run_team_broadcast_command(args) -> int:
    workspace_root = Path(args.workspace).resolve()
    runtime = PersistentTeamRuntime((workspace_root / args.team_dir).resolve())
    runtime.broadcast(args.message, sender=args.sender)
    print('ok')
    return 0


def run_team_shutdown_command(args) -> int:
    workspace_root = Path(args.workspace).resolve()
    runtime = PersistentTeamRuntime((workspace_root / args.team_dir).resolve())
    if args.all:
        runtime.shutdown_all(sender=args.sender, timeout_s=args.timeout_s)
    else:
        runtime.shutdown_teammate(args.to, sender=args.sender)
    print('ok')
    return 0
