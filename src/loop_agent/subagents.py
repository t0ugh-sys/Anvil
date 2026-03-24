from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from .coding_agent import DeciderFn, run_coding_agent
from .core.serialization import run_result_to_dict
from .core.types import StopConfig
from .mailbox import JsonlMailbox, MailMessage
from .task_graph import Task, TaskGraph, TaskStatus


@dataclass(frozen=True)
class SubAgentSpec:
    agent_id: str
    role: str
    workspace_root: Path
    skills: Tuple[str, ...] = tuple()
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SubAgentResult:
    agent_id: str
    task_id: str
    success: bool
    stop_reason: str
    final_output: str
    payload: Dict[str, Any]


class SubAgentRuntime:
    def __init__(self, *, mailbox: JsonlMailbox, task_graph: TaskGraph, coordinator_id: str = 'coordinator') -> None:
        self.mailbox = mailbox
        self.task_graph = task_graph
        self.coordinator_id = coordinator_id

    def spawn(self, spec: SubAgentSpec, task: Task) -> Task:
        assigned = self.task_graph.assign_task(task.id, spec.agent_id)
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

        result = run_coding_agent(
            goal=task.goal,
            decider=decider,
            workspace_root=spec.workspace_root,
            stop=stop or StopConfig(max_steps=8, max_elapsed_s=60.0),
        )
        payload = run_result_to_dict(result, include_history=True)
        if result.done:
            self.task_graph.mark_completed(task_id, metadata={'stop_reason': result.stop_reason.value})
        else:
            self.task_graph.mark_failed(task_id, metadata={'stop_reason': result.stop_reason.value, 'error': result.error})

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
