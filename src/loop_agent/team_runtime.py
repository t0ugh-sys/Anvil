from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .core.types import StopConfig
from .mailbox import JsonlMailbox, MailMessage
from .task_graph import Task, TaskGraph, TaskStatus


@dataclass(frozen=True)
class PersistentTeammateSpec:
    """Specification for spawning a persistent teammate."""

    name: str
    role: str
    workspace_root: Path
    decider: Any  # DeciderFn - function type requires forward reference
    stop: StopConfig
    skills: Tuple[str, ...] = tuple()


@dataclass
class TeamConfig:
    """Configuration for a team runtime."""

    members: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {'members': self.members}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'TeamConfig':
        return cls(members=payload.get('members', []))


class TeamConfigStore:
    """Persistent storage for team configuration."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.config_file = root_dir / 'config.json'
        self._config: TeamConfig | None = None

    def load(self) -> TeamConfig:
        if self._config is not None:
            return self._config
        if self.config_file.exists():
            with self.config_file.open(encoding='utf-8') as f:
                self._config = TeamConfig.from_dict(json.load(f))
        else:
            self._config = TeamConfig()
        return self._config

    def save(self, config: TeamConfig) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        with self.config_file.open('w', encoding='utf-8') as f:
            json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)
        self._config = config


class TaskGraphStore:
    """Persistent storage for task graph."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.graph_file = root_dir / 'tasks.json'

    def load(self) -> TaskGraph:
        if self.graph_file.exists():
            with self.graph_file.open(encoding='utf-8') as f:
                return TaskGraph.from_dict(json.load(f))
        return TaskGraph()

    def save(self, graph: TaskGraph) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        with self.graph_file.open('w', encoding='utf-8') as f:
            json.dump(graph.to_dict(), f, ensure_ascii=False, indent=2)


class PersistentInboxStore:
    """Persistent inbox storage for team messages."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.mailbox = JsonlMailbox(root_dir / 'inbox')

    def send(self, recipient: str, body: str, sender: str, subject: str = '') -> MailMessage:
        message = MailMessage(
            id=str(uuid.uuid4()),
            sender=sender,
            recipient=recipient,
            subject=subject,
            body=body,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.mailbox.send(message)
        return message

    def drain(self, recipient: str) -> Tuple[MailMessage, ...]:
        messages = self.mailbox.inbox(recipient)
        return messages

    def broadcast(self, body: str, sender: str, recipients: List[str]) -> List[MailMessage]:
        messages = []
        for recipient in recipients:
            messages.append(self.send(recipient, body, sender))
        return messages


class PersistentTeamRuntime:
    """Persistent runtime for managing teammate loops.

    Manages teammates backed by .team/config.json and .team/inbox/*.jsonl.
    """

    def __init__(self, team_root: Path) -> None:
        self.team_root = team_root
        self.team_root.mkdir(parents=True, exist_ok=True)
        self.config_store = TeamConfigStore(team_root)
        self.inbox_store = PersistentInboxStore(team_root)
        self.task_graph_store = TaskGraphStore(team_root)
        self._teammates: Dict[str, Any] = {}  # name -> teammate info
        self._shutdown_requests: Dict[str, bool] = {}  # name -> shutdown flag

    def spawn_teammate(self, spec: PersistentTeammateSpec) -> None:
        """Spawn a persistent teammate based on the specification."""
        config = self.config_store.load()
        config.members.append({
            'name': spec.name,
            'role': spec.role,
            'skills': list(spec.skills),
        })
        self.config_store.save(config)
        self._teammates[spec.name] = {
            'spec': spec,
            'active': True,
        }

    def send_message(self, recipient: str, body: str, sender: str, subject: str = '') -> MailMessage:
        """Send a message to a teammate."""
        return self.inbox_store.send(recipient, body, sender, subject)

    def broadcast(self, body: str, sender: str) -> List[MailMessage]:
        """Broadcast a message to all teammates."""
        config = self.config_store.load()
        recipients = [member['name'] for member in config.members]
        return self.inbox_store.broadcast(body, sender, recipients)

    def has_active_tasks(self) -> bool:
        """Check if there are any active (pending or running) tasks."""
        graph = self.load_task_graph()
        for task in graph.tasks():
            if task.status in {TaskStatus.pending, TaskStatus.ready, TaskStatus.running}:
                return True
        return False

    def has_pending_member_messages(self) -> bool:
        """Check if any teammates have pending messages."""
        config = self.config_store.load()
        for member in config.members:
            inbox = self.inbox_store.drain(member['name'])
            if inbox:  # If there are messages, put them back and return True
                # Re-send messages back to inbox
                for msg in inbox:
                    self.inbox_store.send(msg.recipient, msg.body, msg.sender, msg.subject)
                return True
        return False

    def shutdown_teammate(self, name: str, sender: str) -> None:
        """Request shutdown of a specific teammate."""
        self._shutdown_requests[name] = True
        self.send_message(name, 'shutdown', sender, 'shutdown request')

    def shutdown_all(self, sender: str, timeout_s: float = 5.0) -> None:
        """Request shutdown of all teammates."""
        config = self.config_store.load()
        for member in config.members:
            self.shutdown_teammate(member['name'], sender)

    def all_teammates_shutdown(self) -> bool:
        """Check if all teammates have responded to shutdown requests."""
        config = self.config_store.load()
        for member in config.members:
            name = member['name']
            if name in self._teammates and self._teammates[name].get('active', True):
                return False
        return True

    def load_task_graph(self) -> TaskGraph:
        """Load the current task graph."""
        return self.task_graph_store.load()

    def replace_task_graph(self, tasks: List[Task]) -> None:
        """Replace the entire task graph."""
        graph = TaskGraph(tasks)
        self.task_graph_store.save(graph)

    def add_task(self, task: Task) -> TaskGraph:
        """Add a task to the graph and return updated graph."""
        graph = self.load_task_graph()
        graph.add_task(task)
        self.task_graph_store.save(graph)
        return graph

    def dispatch_ready_tasks(self, sender: str) -> None:
        """Dispatch all ready tasks to appropriate teammates."""
        graph = self.load_task_graph()
        ready_tasks = graph.ready_tasks()
        for task in ready_tasks:
            if task.assignee:
                self.send_message(
                    task.assignee,
                    f'Task: {task.title}\n\nGoal: {task.goal}',
                    sender,
                    subject=f'Task: {task.id}',
                )
            graph.set_status(task.id, TaskStatus.running)
        self.task_graph_store.save(graph)

    def complete_task(self, task_id: str, metadata: Dict[str, Any] | None = None) -> None:
        """Mark a task as completed."""
        graph = self.load_task_graph()
        graph.mark_completed(task_id, metadata=metadata)
        self.task_graph_store.save(graph)

    def fail_task(self, task_id: str, metadata: Dict[str, Any] | None = None) -> None:
        """Mark a task as failed."""
        graph = self.load_task_graph()
        graph.mark_failed(task_id, metadata=metadata)
        self.task_graph_store.save(graph)
