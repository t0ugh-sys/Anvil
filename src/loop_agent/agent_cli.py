from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .agent_protocol import render_agent_step_schema
from .coding_agent import run_coding_agent
from .compression import CompressionConfig, summarize_entries_deterministically
from .core.serialization import run_result_to_dict
from .core.types import ContextSnapshot, ObserverFn, StopConfig
from .llm.providers import build_invoke_from_args
from .memory.jsonl_store import JsonlMemoryStore
from .ops.doctor import format_doctor_report, run_provider_doctor
from .run_recorder import RunRecorder
from .skills import SkillLoader, list_skills, get_skill
from .task_graph import Task
from .task_store import TaskStore
from .team_runtime import PersistentTeamRuntime, PersistentTeammateSpec
from .tool_use_loop import DeciderFn
from .tools import build_default_tools


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def _resolve_goal(args: argparse.Namespace) -> str:
    if args.goal_file:
        return Path(args.goal_file).read_text(encoding='utf-8-sig').strip()
    return args.goal.strip()


def _build_coding_decider(args: argparse.Namespace, skills: SkillLoader | None = None) -> DeciderFn:
    from .tool_use_loop import DeciderFn

    invoke = build_invoke_from_args(args, mode='coding')

    def decider(
        goal: str,
        history: Tuple[str, ...],
        tool_results: Tuple[Any, ...],
        state_summary: Dict[str, object],
        last_steps: Tuple[str, ...],
    ) -> str:
        history_window = max(1, args.history_window)
        skill_lines: List[str] = []
        if skills is not None:
            for item in skills.metadata():
                skill_lines.append(f'- {item["name"]}: {item["description"]}')
        prompt = (
            'You are a coding agent. Return strict JSON matching schema.\n'
            'Use tools when needed. Keep a visible todo list updated via the todo_write tool when progress changes.\n'
            + ('Available skills:\n' + '\n'.join(skill_lines) + '\n' if skill_lines else '')
            + 'Do not inline full skill instructions in the prompt. Load them on demand with load_skill.\n'
            + render_agent_step_schema()
            + '\nGoal:\n'
            + goal
            + '\nHistory:\n'
            + str(list(history[-history_window:]))
            + '\nStateSummary:\n'
            + json.dumps(state_summary, ensure_ascii=False)
            + '\nLastSteps:\n'
            + str(list(last_steps))
            + '\nToolResults:\n'
            + str([{'id': r.id, 'ok': r.ok, 'output': r.output[:500], 'error': r.error} for r in tool_results])
            + '\nOnly output JSON.'
        )
        return invoke(prompt)

    return decider


def _build_coding_summarizer(args: argparse.Namespace) -> Optional[SummarizerFn]:
    from .compression import TranscriptEntry
    from .tool_use_loop import SummarizerFn

    if str(args.provider) == 'mock':
        return None

    invoke = build_invoke_from_args(args, mode='coding')

    def summarizer(goal: str, previous_summary: str, transcript: Tuple[TranscriptEntry, ...]) -> str:
        transcript_lines = [entry.render_line()[:400] for entry in transcript[-16:]]
        prompt = (
            'Summarize the coding-agent conversation for long-running context compression.\n'
            'Return plain text only.\n'
            'Keep: user goal, constraints, files changed, tool outcomes, unfinished work.\n'
            f'Goal:\n{goal}\n'
            f'Previous summary:\n{previous_summary or "none"}\n'
            'Recent transcript:\n'
            + '\n'.join(transcript_lines)
        )
        response = invoke(prompt).strip()
        if response:
            return response
        return summarize_entries_deterministically(goal=goal, previous_summary=previous_summary, entries=transcript)

    return summarizer


def _load_skills_from_args(args: argparse.Namespace) -> SkillLoader | None:
    """Load skills from command-line arguments."""
    skills_arg = getattr(args, 'skills', None)
    if not skills_arg:
        return None
    
    loader = SkillLoader()
    for skill_name in skills_arg:
        if skill_name == 'all':
            # Load all built-in skills
            for name in list_skills():
                loader.load(name)
        else:
            if not loader.load(skill_name):
                print(f"Warning: Unknown skill '{skill_name}' - skipping")
    return loader


def _build_jsonl_observer(path: str) -> ObserverFn:
    def observer(event: str, payload: Dict[str, Any]) -> None:
        record = {'event': event, 'payload': payload}
        with open(path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(record, ensure_ascii=False))
            file.write('\n')

    return observer


def _merge_observers(observers: List[ObserverFn]) -> Optional[ObserverFn]:
    active = [item for item in observers if item is not None]
    if not active:
        return None

    def merged(event: str, payload: Dict[str, Any]) -> None:
        for observer in active:
            observer(event, payload)

    return merged


def _run_code_command(args: argparse.Namespace) -> int:
    goal = _resolve_goal(args)
    workspace_root = Path(args.workspace).resolve()
    task_store = TaskStore((workspace_root / args.tasks_dir).resolve()) if args.tasks_dir else None
    compression_config = CompressionConfig(
        micro_keep_last_results=args.micro_compact_keep,
        max_context_tokens=args.max_context_tokens,
        recent_transcript_entries=args.recent_transcript_entries,
    )
    transcripts_dir = (workspace_root / args.transcripts_dir).resolve() if args.transcripts_dir else None
    run_id = args.run_id or _default_run_id()
    memory_run_dir = Path(args.memory_dir) / run_id
    memory_store = JsonlMemoryStore(memory_dir=memory_run_dir, summarize_every=args.summarize_every)
    memory_store.on_event('run_started', {'goal': goal, 'strategy': 'coding', 'facts': []})

    recorder: Optional[RunRecorder] = None
    observers: List[ObserverFn] = []
    if args.observer_file:
        observers.append(_build_jsonl_observer(args.observer_file))
    if args.record_run:
        recorder = RunRecorder.create(base_dir=Path(args.runs_dir))
        observers.append(recorder.write_event)
    observers.append(memory_store.on_event)
    observer = _merge_observers(observers)

    def context_provider() -> ContextSnapshot:
        context = memory_store.load_context(goal=goal, last_k_steps=args.history_window)
        return ContextSnapshot(state_summary=context.state_summary, last_steps=context.last_steps)

    skills = _load_skills_from_args(args)
    decider = _build_coding_decider(args, skills)
    summarizer = _build_coding_summarizer(args)
    result = run_coding_agent(
        goal=goal,
        decider=decider,
        workspace_root=workspace_root,
        stop=StopConfig(max_steps=args.max_steps, max_elapsed_s=args.timeout_s),
        observer=observer,
        context_provider=context_provider,
        skills=skills,
        task_store=task_store,
        compression_config=compression_config,
        transcripts_dir=transcripts_dir,
        summarizer=summarizer,
    )
    memory_store.on_event(
        'run_finished',
        {'done': result.done, 'stop_reason': result.stop_reason.value, 'steps': result.steps},
    )
    if recorder is not None:
        recorder.write_summary(run_result_to_dict(result, include_history=True))

    memory_context = memory_store.load_context(goal=goal, last_k_steps=args.history_window)

    payload = run_result_to_dict(result, include_history=args.include_history)
    payload['workspace'] = str(workspace_root)
    payload['provider'] = args.provider
    payload['model'] = args.model
    payload['memory_state'] = memory_context.state_summary
    payload['memory_last_steps'] = list(memory_context.last_steps)
    payload['memory_run_dir'] = str(memory_run_dir)
    if recorder is not None:
        payload['run_dir'] = str(recorder.run_dir)
    if args.output == 'json':
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(f"done: {result.done}")
        print(f"stop_reason: {result.stop_reason.value}")
        print(f"steps: {result.steps}")
        print(f"final_output: {result.final_output}")
        print(f"memory_run_dir: {memory_run_dir}")
        if recorder is not None:
            print(f"run_dir: {recorder.run_dir}")
    return 0 if result.done else 1


def _parse_teammate(value: str) -> Tuple[str, str]:
    raw = value.strip()
    if ':' not in raw:
        raise ValueError('teammate must be NAME:ROLE')
    name, role = raw.split(':', 1)
    name = name.strip()
    role = role.strip()
    if not name or not role:
        raise ValueError('teammate must be NAME:ROLE')
    return name, role


def _parse_team_message(value: str) -> Tuple[str, str]:
    raw = value.strip()
    if '=' not in raw:
        raise ValueError('message must be TARGET=BODY')
    target, body = raw.split('=', 1)
    target = target.strip()
    body = body.strip()
    if not target or not body:
        raise ValueError('message must be TARGET=BODY')
    return target, body


def _run_team_run_command(args: argparse.Namespace) -> int:
    runtime, team_root, teammate_specs = _spawn_team_runtime(args)
    _append_startup_tasks(runtime, args.task, args.sender)

    expected_replies = 0
    task_mode = bool(args.task)
    for item in args.message:
        recipient, body = _parse_team_message(item)
        runtime.send_message(recipient, body, sender=args.sender)
        expected_replies += 1
    for body in args.broadcast:
        runtime.broadcast(body, sender=args.sender)
        expected_replies += len(teammate_specs)

    replies: List[Dict[str, Any]] = []
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


def _spawn_team_runtime(args: argparse.Namespace) -> tuple[PersistentTeamRuntime, Path, list[PersistentTeammateSpec]]:
    workspace_root = Path(args.workspace).resolve()
    team_root = (workspace_root / args.team_dir).resolve()
    runtime = PersistentTeamRuntime(team_root)
    decider = _build_coding_decider(args, _load_skills_from_args(args))
    teammate_specs: List[PersistentTeammateSpec] = []
    for item in args.teammate:
        name, role = _parse_teammate(item)
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


def _append_startup_tasks(runtime: PersistentTeamRuntime, task_bodies: List[str], sender: str) -> None:
    if not task_bodies:
        return
    runtime.replace_task_graph(
        [
            Task(id=f'task_{index}', title=body, goal=body)
            for index, body in enumerate(task_bodies, start=1)
        ]
    )
    runtime.dispatch_ready_tasks(sender=sender)


def _run_team_serve_command(args: argparse.Namespace) -> int:
    runtime, team_root, teammate_specs = _spawn_team_runtime(args)
    _append_startup_tasks(runtime, args.task, args.sender)
    for item in args.message:
        recipient, body = _parse_team_message(item)
        runtime.send_message(recipient, body, sender=args.sender)
    for body in args.broadcast:
        runtime.broadcast(body, sender=args.sender)

    replies: List[Dict[str, Any]] = []
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
        print(f'mode: service')
        print(f'teammates: {[spec.name for spec in teammate_specs]}')
        print(f'replies: {len(replies)}')
    return 0


def _run_team_add_task_command(args: argparse.Namespace) -> int:
    workspace_root = Path(args.workspace).resolve()
    runtime = PersistentTeamRuntime((workspace_root / args.team_dir).resolve())
    metadata: Dict[str, Any] = {}
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


def _run_team_send_command(args: argparse.Namespace) -> int:
    workspace_root = Path(args.workspace).resolve()
    runtime = PersistentTeamRuntime((workspace_root / args.team_dir).resolve())
    runtime.send_message(args.to, args.message, sender=args.sender)
    print('ok')
    return 0


def _run_team_broadcast_command(args: argparse.Namespace) -> int:
    workspace_root = Path(args.workspace).resolve()
    runtime = PersistentTeamRuntime((workspace_root / args.team_dir).resolve())
    runtime.broadcast(args.message, sender=args.sender)
    print('ok')
    return 0


def _run_team_shutdown_command(args: argparse.Namespace) -> int:
    workspace_root = Path(args.workspace).resolve()
    runtime = PersistentTeamRuntime((workspace_root / args.team_dir).resolve())
    if args.all:
        runtime.shutdown_all(sender=args.sender, timeout_s=args.timeout_s)
    else:
        runtime.shutdown_teammate(args.to, sender=args.sender)
    print('ok')
    return 0


def _run_tools_command(_: argparse.Namespace) -> int:
    names = sorted(build_default_tools().keys())
    print('\n'.join(names))
    return 0


def _run_skills_command(_: argparse.Namespace) -> int:
    """List all available skills."""
    skills = list_skills()
    print("Available skills:")
    for name in skills:
        skill = get_skill(name)
        if skill:
            print(f"  - {name}: {skill.description}")
    print("\nUse --skill <name> to load specific skills")
    print("Use --skill all to load all skills")
    return 0


def _run_replay_command(args: argparse.Namespace) -> int:
    events_file = Path(args.events_file)
    if not events_file.exists():
        print(f'events file not found: {events_file}')
        return 1
    print(events_file.read_text(encoding='utf-8'))
    return 0


def _run_doctor_command(args: argparse.Namespace) -> int:
    api_key = os.getenv(args.api_key_env, '').strip()
    payload = run_provider_doctor(
        base_url=args.base_url,
        model=args.model,
        wire_api=args.wire_api,
        timeout_s=args.provider_timeout_s,
        api_key_present=bool(api_key),
        extra_headers=args.provider_header,
    )
    if args.output == 'json':
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(format_doctor_report(payload))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='agent_cli',
        description='Run LoopAgent as a tool-use feedback loop: model decides, tools execute, results feed back.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    code = subparsers.add_parser(
        'code',
        help='run the coding tool-use loop',
        description='Run the coding runtime as a tool-use feedback loop over a workspace.',
        epilog=(
            'Examples:\n'
            '  agent_cli code --goal "inspect README then finish" --workspace . --provider mock --model mock-v3\n'
            '  agent_cli code --goal-file goal.txt --workspace . --provider openai_compatible --model gpt-5.3-codex \\\n'
            '    --base-url https://codex-api.packycode.com/v1 --wire-api responses --output json\n'
            '  agent_cli code --goal "search docs then summarize" --workspace . --skill web_search --skill memory\n'
            '  agent_cli code --goal "fix tests" --workspace . --observer-file events.jsonl --memory-dir .loopagent/runs\n'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    goal_group = code.add_mutually_exclusive_group(required=True)
    goal_group.add_argument('--goal', help='Direct goal text for the tool-use loop')
    goal_group.add_argument('--goal-file', help='UTF-8 goal file for the tool-use loop')

    execution_group = code.add_argument_group('execution')
    execution_group.add_argument('--workspace', default='.', help='Workspace root available to tools')
    execution_group.add_argument('--max-steps', type=int, default=12, help='Maximum tool-use rounds before stopping')
    execution_group.add_argument('--timeout-s', type=float, default=120.0, help='Maximum elapsed seconds for the run')
    execution_group.add_argument('--output', choices=['text', 'json'], default='text')
    execution_group.add_argument('--include-history', action='store_true')

    provider_group = code.add_argument_group('provider')
    provider_group.add_argument(
        '--provider',
        choices=['mock', 'openai_compatible', 'anthropic', 'gemini'],
        default='mock',
    )
    provider_group.add_argument('--model', default='mock-model')
    provider_group.add_argument('--base-url', default='')
    provider_group.add_argument('--wire-api', choices=['chat_completions', 'responses'], default='chat_completions')
    provider_group.add_argument('--api-key-env', default='OPENAI_API_KEY')
    provider_group.add_argument('--temperature', type=float, default=0.2)
    provider_group.add_argument('--provider-timeout-s', type=float, default=60.0)
    provider_group.add_argument('--provider-debug', action='store_true')
    provider_group.add_argument('--fallback-model', action='append', default=[])
    provider_group.add_argument('--max-retries', type=int, default=2)
    provider_group.add_argument('--retry-backoff-s', type=float, default=1.0)
    provider_group.add_argument('--retry-http-code', action='append', type=int, default=[])
    provider_group.add_argument(
        '--provider-header',
        action='append',
        default=[],
        help='provider extra header Key:Value, repeatable',
    )

    memory_group = code.add_argument_group('memory and artifacts')
    memory_group.add_argument('--history-window', type=int, default=8, help='How many recent loop outputs to feed back')
    memory_group.add_argument('--observer-file', help='Write observer events as JSONL')
    memory_group.add_argument('--memory-dir', default='.loopagent/runs', help='Persistent memory root for run state')
    memory_group.add_argument('--run-id', help='Optional run id for memory and artifacts')
    memory_group.add_argument('--summarize-every', type=int, default=5, help='Refresh memory summary every N rounds')
    memory_group.add_argument('--record-run', action='store_true', default=True)
    memory_group.add_argument('--no-record-run', action='store_false', dest='record_run')
    memory_group.add_argument('--runs-dir', default='.loopagent/runs', help='Directory for structured run artifacts')
    memory_group.add_argument('--tasks-dir', default='.tasks', help='Workspace-relative task graph directory injected into context')
    memory_group.add_argument('--transcripts-dir', default='.transcripts', help='Workspace-relative archive directory for compacted transcripts')
    memory_group.add_argument('--max-context-tokens', type=int, default=50000, help='Auto-compact when estimated context exceeds this budget')
    memory_group.add_argument('--micro-compact-keep', type=int, default=3, help='Keep full content for the most recent N tool results')
    memory_group.add_argument('--recent-transcript-entries', type=int, default=8, help='Expose the most recent compacted transcript entries in state summary')

    tool_group = code.add_argument_group('tool dispatch')
    tool_group.add_argument(
        '--skill',
        action='append',
        default=[],
        dest='skills',
        help='Skills to load into the tool dispatch (web_search, memory, files, commands, browser, or "all")',
    )

    tools = subparsers.add_parser(
        'tools',
        help='list tools exposed to the loop',
        description='List tool handlers available to the coding tool-use loop.',
    )
    tools.set_defaults(handler=_run_tools_command)

    skills_parser = subparsers.add_parser(
        'skills',
        help='list available skills',
        description='List optional skills that can extend the loop tool dispatch.',
    )
    skills_parser.set_defaults(handler=_run_skills_command)

    replay = subparsers.add_parser(
        'replay',
        help='print recorded loop events',
        description='Print a recorded JSONL event stream for a previous tool-use run.',
    )
    replay.add_argument('--events-file', required=True)
    replay.set_defaults(handler=_run_replay_command)

    team = subparsers.add_parser(
        'team',
        help='run or control persistent teammates',
        description='Manage persistent teammate loops backed by .team/config.json and .team/inbox/*.jsonl',
    )
    team_subparsers = team.add_subparsers(dest='team_command', required=True)

    team_run = team_subparsers.add_parser(
        'run',
        help='start teammate loops in the current process',
        description='Spawn persistent teammates, deliver messages, and keep threads alive until idle or timeout.',
    )
    team_run.add_argument('--workspace', default='.', help='Workspace root for teammate execution')
    team_run.add_argument('--team-dir', default='.team', help='Workspace-relative team runtime directory')
    team_run.add_argument('--teammate', action='append', default=[], required=True, help='Teammate in NAME:ROLE format')
    team_run.add_argument('--message', action='append', default=[], help='Send startup message TARGET=BODY')
    team_run.add_argument('--broadcast', action='append', default=[], help='Broadcast startup message to all teammates')
    team_run.add_argument('--task', action='append', default=[], help='Create and dispatch one task goal per entry')
    team_run.add_argument('--sender', default='lead')
    team_run.add_argument('--service-timeout-s', type=float, default=5.0)
    team_run.add_argument('--poll-interval-s', type=float, default=0.05)
    team_run.add_argument('--output', choices=['text', 'json'], default='text')
    team_run.add_argument('--max-steps', type=int, default=12)
    team_run.add_argument('--timeout-s', type=float, default=120.0)
    team_run.add_argument(
        '--provider',
        choices=['mock', 'openai_compatible', 'anthropic', 'gemini'],
        default='mock',
    )
    team_run.add_argument('--model', default='mock-model')
    team_run.add_argument('--base-url', default='')
    team_run.add_argument('--wire-api', choices=['chat_completions', 'responses'], default='chat_completions')
    team_run.add_argument('--api-key-env', default='OPENAI_API_KEY')
    team_run.add_argument('--temperature', type=float, default=0.2)
    team_run.add_argument('--provider-timeout-s', type=float, default=60.0)
    team_run.add_argument('--provider-debug', action='store_true')
    team_run.add_argument('--fallback-model', action='append', default=[])
    team_run.add_argument('--max-retries', type=int, default=2)
    team_run.add_argument('--retry-backoff-s', type=float, default=1.0)
    team_run.add_argument('--retry-http-code', action='append', type=int, default=[])
    team_run.add_argument('--provider-header', action='append', default=[])
    team_run.add_argument('--history-window', type=int, default=8)
    team_run.add_argument('--skill', action='append', default=[], dest='skills')
    team_run.set_defaults(handler=_run_team_run_command)

    team_serve = team_subparsers.add_parser(
        'serve',
        help='start a long-lived team service',
        description='Spawn persistent teammates and keep the process alive to serve inbox traffic and task dispatch.',
    )
    team_serve.add_argument('--workspace', default='.', help='Workspace root for teammate execution')
    team_serve.add_argument('--team-dir', default='.team', help='Workspace-relative team runtime directory')
    team_serve.add_argument('--teammate', action='append', default=[], required=True, help='Teammate in NAME:ROLE format')
    team_serve.add_argument('--message', action='append', default=[], help='Send startup message TARGET=BODY')
    team_serve.add_argument('--broadcast', action='append', default=[], help='Broadcast startup message to all teammates')
    team_serve.add_argument('--task', action='append', default=[], help='Create startup task goals before entering service loop')
    team_serve.add_argument('--sender', default='lead')
    team_serve.add_argument('--service-timeout-s', type=float, default=300.0, help='Max wall time for the service; <=0 means no deadline')
    team_serve.add_argument('--idle-exit-s', type=float, default=0.0, help='Optional idle timeout before the service exits')
    team_serve.add_argument('--poll-interval-s', type=float, default=0.05)
    team_serve.add_argument('--output', choices=['text', 'json'], default='text')
    team_serve.add_argument('--max-steps', type=int, default=12)
    team_serve.add_argument('--timeout-s', type=float, default=120.0)
    team_serve.add_argument(
        '--provider',
        choices=['mock', 'openai_compatible', 'anthropic', 'gemini'],
        default='mock',
    )
    team_serve.add_argument('--model', default='mock-model')
    team_serve.add_argument('--base-url', default='')
    team_serve.add_argument('--wire-api', choices=['chat_completions', 'responses'], default='chat_completions')
    team_serve.add_argument('--api-key-env', default='OPENAI_API_KEY')
    team_serve.add_argument('--temperature', type=float, default=0.2)
    team_serve.add_argument('--provider-timeout-s', type=float, default=60.0)
    team_serve.add_argument('--provider-debug', action='store_true')
    team_serve.add_argument('--fallback-model', action='append', default=[])
    team_serve.add_argument('--max-retries', type=int, default=2)
    team_serve.add_argument('--retry-backoff-s', type=float, default=1.0)
    team_serve.add_argument('--retry-http-code', action='append', type=int, default=[])
    team_serve.add_argument('--provider-header', action='append', default=[])
    team_serve.add_argument('--history-window', type=int, default=8)
    team_serve.add_argument('--skill', action='append', default=[], dest='skills')
    team_serve.set_defaults(handler=_run_team_serve_command)

    team_send = team_subparsers.add_parser('send', help='append one message to a teammate inbox')
    team_send.add_argument('--workspace', default='.')
    team_send.add_argument('--team-dir', default='.team')
    team_send.add_argument('--to', required=True)
    team_send.add_argument('--message', required=True)
    team_send.add_argument('--sender', default='lead')
    team_send.set_defaults(handler=_run_team_send_command)

    team_broadcast = team_subparsers.add_parser('broadcast', help='append one broadcast message to all teammate inboxes')
    team_broadcast.add_argument('--workspace', default='.')
    team_broadcast.add_argument('--team-dir', default='.team')
    team_broadcast.add_argument('--message', required=True)
    team_broadcast.add_argument('--sender', default='lead')
    team_broadcast.set_defaults(handler=_run_team_broadcast_command)

    team_shutdown = team_subparsers.add_parser('shutdown', help='request teammate shutdown')
    team_shutdown.add_argument('--workspace', default='.')
    team_shutdown.add_argument('--team-dir', default='.team')
    team_shutdown.add_argument('--sender', default='lead')
    team_shutdown.add_argument('--timeout-s', type=float, default=5.0)
    target_group = team_shutdown.add_mutually_exclusive_group(required=True)
    target_group.add_argument('--to')
    target_group.add_argument('--all', action='store_true')
    team_shutdown.set_defaults(handler=_run_team_shutdown_command)

    team_add_task = team_subparsers.add_parser('add-task', help='append one task into the team task graph')
    team_add_task.add_argument('--workspace', default='.')
    team_add_task.add_argument('--team-dir', default='.team')
    team_add_task.add_argument('--goal', required=True)
    team_add_task.add_argument('--title')
    team_add_task.add_argument('--task-id')
    team_add_task.add_argument('--depends-on', action='append', default=[])
    team_add_task.add_argument('--assignee')
    team_add_task.add_argument('--role')
    team_add_task.add_argument('--sender', default='lead')
    team_add_task.add_argument('--output', choices=['text', 'json'], default='text')
    team_add_task.set_defaults(handler=_run_team_add_task_command)

    doctor = subparsers.add_parser(
        'doctor',
        help='diagnose provider connectivity',
        description='Diagnose provider connectivity before running the tool-use loop.',
    )
    doctor.add_argument('--provider', choices=['openai_compatible'], default='openai_compatible')
    doctor.add_argument('--model', default='gpt-5.3-codex')
    doctor.add_argument('--base-url', required=True)
    doctor.add_argument('--wire-api', choices=['chat_completions', 'responses'], default='responses')
    doctor.add_argument('--api-key-env', default='OPENAI_API_KEY')
    doctor.add_argument('--provider-timeout-s', type=float, default=20.0)
    doctor.add_argument('--provider-header', action='append', default=[])
    doctor.add_argument('--output', choices=['text', 'json'], default='text')
    doctor.set_defaults(handler=_run_doctor_command)

    code.set_defaults(handler=_run_code_command)
    return parser


def main() -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[call-arg]
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[call-arg]
    parser = build_parser()
    args = parser.parse_args()
    code = args.handler(args)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
