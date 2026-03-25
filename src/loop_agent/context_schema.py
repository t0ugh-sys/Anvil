from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from .mailbox import JsonlMailbox
from .policies import ToolPolicy
from .task_graph import TaskGraph, TaskStatus


@dataclass(frozen=True)
class OrchestrationContextInput:
    goal: str
    agent_id: str
    current_task_id: str | None
    workspace_root: Path
    mailbox: JsonlMailbox | None = None
    task_graph: TaskGraph | None = None
    policy: ToolPolicy | None = None
    facts: Tuple[str, ...] = tuple()
    current_plan: Tuple[str, ...] = tuple()
    recent_steps: Tuple[str, ...] = tuple()
    memory_summary: Dict[str, Any] | None = None
    isolation_mode: str | None = None


def build_orchestration_context(payload: OrchestrationContextInput) -> Dict[str, Any]:
    task_state = _build_task_state(payload.task_graph, payload.current_task_id)
    mailbox_digest = _build_mailbox_digest(payload.mailbox, payload.agent_id)
    return {
        'context_schema': 'orchestration-v1',
        'goal': payload.goal,
        'agent': {
            'agent_id': payload.agent_id,
            'current_task_id': payload.current_task_id,
        },
        'workspace': {
            'root': str(payload.workspace_root),
            'isolation_mode': payload.isolation_mode or 'none',
        },
        'task_state': task_state,
        'mailbox_digest': mailbox_digest,
        'policy': (payload.policy or ToolPolicy.allow_all()).to_dict(),
        'durable_facts': list(payload.facts),
        'current_plan': list(payload.current_plan),
        'recent_steps': list(payload.recent_steps),
        'memory_summary': payload.memory_summary or {},
    }


def _build_task_state(task_graph: TaskGraph | None, current_task_id: str | None) -> Dict[str, Any]:
    if task_graph is None:
        return {
            'current_task_id': current_task_id,
            'ready': [],
            'running': [],
            'blocked': [],
            'completed': [],
            'failed': [],
        }

    tasks = task_graph.tasks()
    return {
        'current_task_id': current_task_id,
        'ready': [task.id for task in tasks if task.status == TaskStatus.ready],
        'running': [task.id for task in tasks if task.status == TaskStatus.running],
        'blocked': [task.id for task in tasks if task.status == TaskStatus.blocked],
        'completed': [task.id for task in tasks if task.status == TaskStatus.completed],
        'failed': [task.id for task in tasks if task.status == TaskStatus.failed],
    }


def _build_mailbox_digest(mailbox: JsonlMailbox | None, recipient: str) -> Dict[str, Any]:
    if mailbox is None:
        return {'recipient': recipient, 'count': 0, 'subjects': [], 'task_ids': []}
    return mailbox.summary_for(recipient)
