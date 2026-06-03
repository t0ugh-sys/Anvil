from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from .coding_agent import DeciderFn, run_coding_agent
from .context_schema import OrchestrationContextInput, build_orchestration_context
from .core.serialization import run_result_to_dict
from .core.types import ContextSnapshot, StopConfig
from .mailbox import JsonlMailbox, MailMessage
from .policies import ToolPolicy
from .task_graph import Task, TaskGraph, TaskStatus
from .task_store import TaskStore
from .worktree_manager import WorktreeManager

__all__ = ['SubAgentResult', 'SubAgentRuntime', 'SubAgentSpec', 'TaskNotification']


@dataclass(frozen=True)
class TaskNotification:
    """Structured task notification — mirrors Claude Code's <task-notification> XML.

    Wraps subagent results with usage stats for the parent agent.
    """
    task_id: str
    agent_id: str
    status: str  # 'completed' | 'failed' | 'killed'
    summary: str
    result: str
    total_tokens: int = 0
    tool_uses: int = 0
    duration_ms: int = 0

    def to_xml(self) -> str:
        """Render as XML for injection into parent context."""
        return (
            f'<task-notification>\n'
            f'  <task-id>{self.task_id}</task-id>\n'
            f'  <status>{self.status}</status>\n'
            f'  <summary>{self.summary}</summary>\n'
            f'  <result>{self.result[:2000]}</result>\n'
            f'  <usage>\n'
            f'    <total_tokens>{self.total_tokens}</total_tokens>\n'
            f'    <tool_uses>{self.tool_uses}</tool_uses>\n'
            f'    <duration_ms>{self.duration_ms}</duration_ms>\n'
            f'  </usage>\n'
            f'</task-notification>'
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'agent_id': self.agent_id,
            'status': self.status,
            'summary': self.summary,
            'result': self.result[:2000],
            'total_tokens': self.total_tokens,
            'tool_uses': self.tool_uses,
            'duration_ms': self.duration_ms,
        }


@dataclass(frozen=True)
class SubAgentSpec:
    agent_id: str
    role: str
    workspace_root: Path
    skills: Tuple[str, ...] = tuple()
    policy: ToolPolicy = ToolPolicy.allow_all()
    metadata: Dict[str, Any] = field(default_factory=dict)
    model: str | None = None  # Model override (e.g. 'haiku' for fast tasks)


@dataclass(frozen=True)
class SubAgentResult:
    agent_id: str
    task_id: str
    success: bool
    stop_reason: str
    final_output: str
    payload: Dict[str, Any]
    duration_ms: int = 0

    def build_notification(self) -> TaskNotification:
        """Build a structured TaskNotification from this result."""
        status = 'completed' if self.success else 'failed'
        history = self.payload.get('history', [])
        tool_uses = sum(1 for h in history if isinstance(h, str) and h.startswith('tool['))
        return TaskNotification(
            task_id=self.task_id,
            agent_id=self.agent_id,
            status=status,
            summary=f'Task {self.task_id} {status}: {self.stop_reason}',
            result=self.final_output,
            tool_uses=tool_uses,
            duration_ms=self.duration_ms,
        )


class SubAgentRuntime:
    def __init__(
        self,
        *,
        mailbox: JsonlMailbox,
        task_graph: TaskGraph,
        coordinator_id: str = 'coordinator',
        worktree_manager: WorktreeManager | None = None,
        task_store: TaskStore | None = None,
    ) -> None:
        self.mailbox = mailbox
        self.task_graph = task_graph
        self.coordinator_id = coordinator_id
        self.worktree_manager = worktree_manager
        self.task_store = task_store
        self._persist_task_graph()

    def _persist_task_graph(self) -> None:
        if self.task_store is None:
            return
        self.task_store.save_graph(self.task_graph)

    def spawn(self, spec: SubAgentSpec, task: Task) -> Task:
        assigned = self.task_graph.assign_task(task.id, spec.agent_id)
        self._persist_task_graph()
        self.mailbox.send(
            MailMessage(
                id=f'{task.id}:assigned',
                sender=self.coordinator_id,
                recipient=spec.agent_id,
                subject=f'Assigned task {task.id}',
                body=task.goal,
                task_id=task.id,
                metadata={'role': spec.role, 'skills': list(spec.skills)},
            )
        )
        return assigned

    def run_once(
        self,
        *,
        spec: SubAgentSpec,
        task_id: str,
        decider: DeciderFn,
        stop: StopConfig | None = None,
    ) -> SubAgentResult:
        task = self.task_graph.get_task(task_id)
        self.task_graph.mark_running(task_id, metadata={'agent_id': spec.agent_id})
        self._persist_task_graph()
        self.mailbox.send(
            MailMessage(
                id=f'{task_id}:started',
                sender=spec.agent_id,
                recipient=self.coordinator_id,
                subject=f'Started task {task_id}',
                body=task.goal,
                task_id=task_id,
            )
        )

        lease = self.worktree_manager.create(task_id) if self.worktree_manager is not None else None
        workspace_root = lease.workspace_path if lease is not None else spec.workspace_root
        context_snapshot = build_orchestration_context(
            OrchestrationContextInput(
                goal=task.goal,
                agent_id=spec.agent_id,
                current_task_id=task_id,
                workspace_root=workspace_root,
                mailbox=self.mailbox,
                task_graph=self.task_graph,
                policy=spec.policy,
                facts=tuple(
                    item for item in task.metadata.get('facts', [])
                    if isinstance(item, str)
                ),
                current_plan=tuple(
                    item for item in task.metadata.get('current_plan', [])
                    if isinstance(item, str)
                ),
                isolation_mode=lease.mode if lease is not None else 'none',
            )
        )
        start_ms = int(_time.time() * 1000)
        try:
            result = run_coding_agent(
                goal=task.goal,
                decider=decider,
                workspace_root=workspace_root,
                stop=stop or StopConfig(max_steps=8, max_elapsed_s=60.0),
                context_provider=lambda: ContextSnapshot(
                    state_summary=context_snapshot,
                    last_steps=tuple(str(item) for item in context_snapshot.get('recent_steps', [])),
                ),
                policy=spec.policy,
            )
        finally:
            if lease is not None:
                self.worktree_manager.cleanup(lease)
        duration_ms = int(_time.time() * 1000) - start_ms
        payload = run_result_to_dict(result, include_history=True)
        if result.done:
            self.task_graph.mark_completed(
                task_id,
                metadata={'stop_reason': result.stop_reason.value, 'workspace_root': str(workspace_root)},
            )
        else:
            self.task_graph.mark_failed(
                task_id,
                metadata={
                    'stop_reason': result.stop_reason.value,
                    'error': result.error,
                    'workspace_root': str(workspace_root),
                },
            )
        self._persist_task_graph()

        self.mailbox.send(
            MailMessage(
                id=f'{task_id}:finished',
                sender=spec.agent_id,
                recipient=self.coordinator_id,
                subject=f'Finished task {task_id}',
                body=result.final_output,
                task_id=task_id,
                metadata={'done': result.done, 'stop_reason': result.stop_reason.value},
            )
        )
        return SubAgentResult(
            agent_id=spec.agent_id,
            task_id=task_id,
            success=result.done,
            stop_reason=result.stop_reason.value,
            final_output=result.final_output,
            payload=payload,
            duration_ms=duration_ms,
        )

    def dispatch_ready_tasks(
        self,
        *,
        specs: Iterable[SubAgentSpec],
        decider: DeciderFn,
        stop: StopConfig | None = None,
    ) -> Tuple[SubAgentResult, ...]:
        ready_tasks = list(self.task_graph.ready_tasks())
        agents = list(specs)
        results: list[SubAgentResult] = []
        for task, spec in zip(ready_tasks, agents):
            self.spawn(spec, task)
            results.append(self.run_once(spec=spec, task_id=task.id, decider=decider, stop=stop))
        return tuple(results)
