from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from .coding_agent import DeciderFn, run_coding_agent
from .core.serialization import run_result_to_dict
from .core.types import StopConfig
from .policies import ToolPolicy
from .skills import SkillLoader
from .task_graph import Task, TaskGraph, TaskStatus
from .task_store import TaskStore


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TeamMessageType(str, Enum):
    message = 'message'
    broadcast = 'broadcast'
    shutdown_request = 'shutdown_request'
    shutdown_response = 'shutdown_response'
    plan_approval_response = 'plan_approval_response'


@dataclass(frozen=True)
class TeamMessage:
    id: str
    sender: str
    recipient: str
    message_type: TeamMessageType
    body: str
    created_at: str = ''
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'sender': self.sender,
            'recipient': self.recipient,
            'message_type': self.message_type.value,
            'body': self.body,
            'created_at': self.created_at or _utc_now(),
            'metadata': dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'TeamMessage':
        return cls(
            id=str(payload.get('id', '')).strip(),
            sender=str(payload.get('sender', '')).strip(),
            recipient=str(payload.get('recipient', '')).strip(),
            message_type=TeamMessageType(str(payload.get('message_type', TeamMessageType.message.value)).strip()),
            body=str(payload.get('body', '')),
            created_at=str(payload.get('created_at', '')),
            metadata=dict(payload.get('metadata', {})) if isinstance(payload.get('metadata', {}), dict) else {},
        )


@dataclass(frozen=True)
class TeamMember:
    name: str
    role: str
    status: str = 'idle'
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'role': self.role,
            'status': self.status,
            'metadata': dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'TeamMember':
        return cls(
            name=str(payload.get('name', '')).strip(),
            role=str(payload.get('role', '')).strip(),
            status=str(payload.get('status', 'idle')).strip() or 'idle',
            metadata=dict(payload.get('metadata', {})) if isinstance(payload.get('metadata', {}), dict) else {},
        )


@dataclass(frozen=True)
class TeamConfig:
    team_name: str = 'default'
    members: Tuple[TeamMember, ...] = tuple()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'team_name': self.team_name,
            'members': [member.to_dict() for member in self.members],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'TeamConfig':
        raw_members = payload.get('members', [])
        members = tuple(
            TeamMember.from_dict(item)
            for item in raw_members
            if isinstance(item, dict)
        ) if isinstance(raw_members, list) else tuple()
        return cls(
            team_name=str(payload.get('team_name', 'default')).strip() or 'default',
            members=members,
        )


class TeamConfigStore:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.root_dir / 'config.json'
        self._lock = threading.RLock()
        if not self.config_file.exists():
            self.save(TeamConfig())

    def load(self) -> TeamConfig:
        with self._lock:
            if not self.config_file.exists():
                return TeamConfig()
            text = self.config_file.read_text(encoding='utf-8').strip()
            if not text:
                return TeamConfig()
            payload = json.loads(text)
            return TeamConfig.from_dict(payload if isinstance(payload, dict) else {})

    def save(self, config: TeamConfig) -> None:
        with self._lock:
            self._write_config(config)

    def upsert_member(self, member: TeamMember) -> TeamConfig:
        with self._lock:
            config = self.load()
            members = {item.name: item for item in config.members}
            members[member.name] = member
            updated = TeamConfig(team_name=config.team_name, members=tuple(sorted(members.values(), key=lambda item: item.name)))
            self._write_config(updated)
            return updated

    def update_member_status(self, name: str, status: str) -> TeamConfig:
        config = self.load()
        members = []
        found = False
        for member in config.members:
            if member.name == name:
                members.append(TeamMember(name=member.name, role=member.role, status=status, metadata=member.metadata))
                found = True
            else:
                members.append(member)
        if not found:
            raise ValueError(f'unknown teammate: {name}')
        updated = TeamConfig(team_name=config.team_name, members=tuple(members))
        self.save(updated)
        return updated

    def _write_config(self, config: TeamConfig) -> None:
        self.config_file.write_text(
            json.dumps(config.to_dict(), ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    def member_names(self) -> Tuple[str, ...]:
        return tuple(member.name for member in self.load().members)


class JsonlTeamInboxStore:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def inbox_file(self, recipient: str) -> Path:
        return self.root_dir / f'{recipient}.jsonl'

    def send(self, message: TeamMessage) -> None:
        payload = message.to_dict()
        with self._lock:
            with self.inbox_file(message.recipient).open('a', encoding='utf-8') as file:
                file.write(json.dumps(payload, ensure_ascii=False))
                file.write('\n')

    def drain(self, recipient: str) -> Tuple[TeamMessage, ...]:
        path = self.inbox_file(recipient)
        with self._lock:
            if not path.exists():
                return tuple()
            rows = []
            with path.open('r', encoding='utf-8') as file:
                for line in file:
                    text = line.strip()
                    if not text:
                        continue
                    payload = json.loads(text)
                    if isinstance(payload, dict):
                        rows.append(TeamMessage.from_dict(payload))
            path.write_text('', encoding='utf-8')
        return tuple(rows)

    def peek(self, recipient: str) -> Tuple[TeamMessage, ...]:
        path = self.inbox_file(recipient)
        if not path.exists():
            return tuple()
        rows = []
        with path.open('r', encoding='utf-8') as file:
            for line in file:
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                if isinstance(payload, dict):
                    rows.append(TeamMessage.from_dict(payload))
        return tuple(rows)

    def has_messages(self, recipient: str) -> bool:
        path = self.inbox_file(recipient)
        if not path.exists():
            return False
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                return True
        return False


@dataclass(frozen=True)
class PersistentTeammateSpec:
    name: str
    role: str
    workspace_root: Path
    decider: DeciderFn
    stop: StopConfig = StopConfig(max_steps=6, max_elapsed_s=60.0)
    policy: ToolPolicy = ToolPolicy.allow_all()
    skills: Tuple[str, ...] = tuple()
    metadata: Dict[str, Any] = field(default_factory=dict)


class PersistentTeamRuntime:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.config_store = TeamConfigStore(self.root_dir)
        self.inbox_store = JsonlTeamInboxStore(self.root_dir / 'inbox')
        self.task_store = TaskStore(self.root_dir / 'tasks')
        self._threads: Dict[str, threading.Thread] = {}
        self._stop_events: Dict[str, threading.Event] = {}
        self._specs: Dict[str, PersistentTeammateSpec] = {}
        self._lock = threading.Lock()

    def spawn_teammate(self, spec: PersistentTeammateSpec) -> None:
        with self._lock:
            if spec.name in self._threads:
                raise ValueError(f'teammate already exists: {spec.name}')
            self.config_store.upsert_member(
                TeamMember(name=spec.name, role=spec.role, status='idle', metadata=spec.metadata)
            )
            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._run_teammate_loop,
                args=(spec, stop_event),
                name=f'teammate-{spec.name}',
                daemon=True,
            )
            self._stop_events[spec.name] = stop_event
            self._specs[spec.name] = spec
            self._threads[spec.name] = thread
            thread.start()

    def send_message(
        self,
        recipient: str,
        body: str,
        *,
        sender: str = 'lead',
        message_type: TeamMessageType = TeamMessageType.message,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        self.inbox_store.send(
            TeamMessage(
                id=uuid.uuid4().hex,
                sender=sender,
                recipient=recipient,
                message_type=message_type,
                body=body,
                metadata=metadata or {},
            )
        )

    def broadcast(
        self,
        body: str,
        *,
        sender: str = 'lead',
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        for recipient in self.config_store.member_names():
            if recipient == sender:
                continue
            self.send_message(
                recipient,
                body,
                sender=sender,
                message_type=TeamMessageType.broadcast,
                metadata=metadata,
            )

    def shutdown_teammate(self, name: str, *, sender: str = 'lead') -> None:
        self.send_message(
            name,
            'shutdown requested',
            sender=sender,
            message_type=TeamMessageType.shutdown_request,
        )

    def shutdown_all(self, *, sender: str = 'lead', timeout_s: float = 5.0) -> None:
        for name in list(self._threads.keys()):
            self.shutdown_teammate(name, sender=sender)
        deadline = time.time() + timeout_s
        for name, thread in list(self._threads.items()):
            remaining = max(0.0, deadline - time.time())
            thread.join(timeout=remaining)

    def teammate_status(self, name: str) -> str:
        for member in self.config_store.load().members:
            if member.name == name:
                return member.status
        raise ValueError(f'unknown teammate: {name}')

    def replace_task_graph(self, tasks: Iterable[Task]) -> TaskGraph:
        return self.task_store.replace_graph(tasks)

    def load_task_graph(self) -> TaskGraph:
        return self.task_store.load_graph()

    def add_task(self, task: Task) -> TaskGraph:
        with self._lock:
            graph = self.task_store.load_graph()
            graph.add_task(task)
            self.task_store.save_graph(graph)
            return graph

    def has_active_tasks(self) -> bool:
        graph = self.task_store.load_graph()
        return any(task.status in {TaskStatus.pending, TaskStatus.ready, TaskStatus.running} for task in graph.tasks())

    def has_pending_member_messages(self) -> bool:
        for recipient in self.config_store.member_names():
            if self.inbox_store.has_messages(recipient):
                return True
        return False

    def all_teammates_shutdown(self) -> bool:
        members = self.config_store.load().members
        return bool(members) and all(member.status == 'shutdown' for member in members)

    def dispatch_ready_tasks(self, *, sender: str = 'lead') -> Tuple[str, ...]:
        with self._lock:
            try:
                graph = self.task_store.load_graph()
            except Exception:
                return tuple()
            if not graph.tasks():
                return tuple()

            members = {
                member.name: member
                for member in self.config_store.load().members
                if member.status == 'idle'
            }
            assigned: list[str] = []
            for task in graph.ready_tasks():
                teammate_name = self._pick_teammate_for_task(task, members)
                if teammate_name is None:
                    continue
                graph.assign_task(task.id, teammate_name)
                graph.mark_running(task.id, metadata={'assigned_by': sender})
                members.pop(teammate_name, None)
                self.inbox_store.send(
                    TeamMessage(
                        id=uuid.uuid4().hex,
                        sender=sender,
                        recipient=teammate_name,
                        message_type=TeamMessageType.message,
                        body=task.goal,
                        metadata={'task_id': task.id, 'task_title': task.title},
                    )
                )
                assigned.append(task.id)
            if assigned:
                self.task_store.save_graph(graph)
            return tuple(assigned)

    def _pick_teammate_for_task(self, task: Task, members: Dict[str, TeamMember]) -> str | None:
        if task.assignee and task.assignee in members:
            return task.assignee
        role_hint = str(task.metadata.get('role', '')).strip()
        if role_hint:
            for member in members.values():
                if member.role == role_hint:
                    return member.name
        return next(iter(members.keys()), None)

    def _run_teammate_loop(self, spec: PersistentTeammateSpec, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            messages = self.inbox_store.drain(spec.name)
            if not messages:
                time.sleep(0.05)
                continue
            for message in messages:
                if message.message_type == TeamMessageType.shutdown_request:
                    self.config_store.update_member_status(spec.name, 'shutdown')
                    if message.sender:
                        self.inbox_store.send(
                            TeamMessage(
                                id=uuid.uuid4().hex,
                                sender=spec.name,
                                recipient=message.sender,
                                message_type=TeamMessageType.shutdown_response,
                                body='shutdown accepted',
                                metadata={'approved': True},
                            )
                        )
                    stop_event.set()
                    break
                if message.message_type not in {TeamMessageType.message, TeamMessageType.broadcast}:
                    continue
                self.config_store.update_member_status(spec.name, 'working')
                result = run_coding_agent(
                    goal=message.body,
                    decider=spec.decider,
                    workspace_root=spec.workspace_root,
                    stop=spec.stop,
                    policy=spec.policy,
                    skills=_load_skills(spec.skills),
                )
                task_id = str(message.metadata.get('task_id', '')).strip()
                if task_id:
                    self._complete_task(task_id, spec.name, result.done, result)
                self.inbox_store.send(
                    TeamMessage(
                        id=uuid.uuid4().hex,
                        sender=spec.name,
                        recipient=message.sender or 'lead',
                        message_type=TeamMessageType.message,
                        body=result.final_output,
                        metadata={
                            'source_message_id': message.id,
                            'task_id': task_id,
                            'done': result.done,
                            'stop_reason': result.stop_reason.value,
                            'payload': run_result_to_dict(result, include_history=True),
                        },
                    )
                )
                self.config_store.update_member_status(spec.name, 'idle')
                self.dispatch_ready_tasks(sender='scheduler')

    def _complete_task(self, task_id: str, teammate_name: str, done: bool, result) -> None:
        with self._lock:
            graph = self.task_store.load_graph()
            if done:
                graph.mark_completed(
                    task_id,
                    metadata={
                        'agent_id': teammate_name,
                        'stop_reason': result.stop_reason.value,
                        'final_output': result.final_output,
                    },
                )
            else:
                graph.mark_failed(
                    task_id,
                    metadata={
                        'agent_id': teammate_name,
                        'stop_reason': result.stop_reason.value,
                        'error': result.error,
                    },
                )
            self.task_store.save_graph(graph)


def _load_skills(skill_names: Iterable[str]) -> SkillLoader | None:
    names = tuple(name for name in skill_names if name)
    if not names:
        return None
    loader = SkillLoader()
    loaded_any = False
    for name in names:
        loaded_any = loader.load(name) or loaded_any
    return loader if loaded_any else None
