from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from .coding_agent import DeciderFn
from .core.types import StopConfig
from .subagents import SubAgentResult, SubAgentRuntime, SubAgentSpec
from .task_graph import TaskGraph, TaskStatus


@dataclass(frozen=True)
class ScheduleBatchResult:
    iteration: int
    dispatched: Tuple[SubAgentResult, ...]
    pending_count: int
    ready_count: int
    running_count: int
    completed_count: int
    failed_count: int


class TaskScheduler:
    def __init__(self, *, runtime: SubAgentRuntime, max_parallel_agents: int | None = None) -> None:
        self.runtime = runtime
        self.max_parallel_agents = max_parallel_agents

    def run_batch(
        self,
        *,
        specs: Iterable[SubAgentSpec],
        decider: DeciderFn,
        stop: StopConfig | None = None,
        iteration: int = 1,
    ) -> ScheduleBatchResult:
        active_specs = list(specs)
        if self.max_parallel_agents is not None:
            active_specs = active_specs[: self.max_parallel_agents]
        dispatched = self.runtime.dispatch_ready_tasks(specs=active_specs, decider=decider, stop=stop)
        graph = self.runtime.task_graph
        return ScheduleBatchResult(
            iteration=iteration,
            dispatched=dispatched,
            pending_count=_count(graph, TaskStatus.pending),
            ready_count=_count(graph, TaskStatus.ready),
            running_count=_count(graph, TaskStatus.running),
            completed_count=_count(graph, TaskStatus.completed),
            failed_count=_count(graph, TaskStatus.failed),
        )

    def run_until_idle(
        self,
        *,
        specs: Iterable[SubAgentSpec],
        decider: DeciderFn,
        stop: StopConfig | None = None,
        max_rounds: int = 20,
    ) -> Tuple[ScheduleBatchResult, ...]:
        if max_rounds < 1:
            raise ValueError('max_rounds must be >= 1')
        active_specs = tuple(specs)
        results: list[ScheduleBatchResult] = []
        for iteration in range(1, max_rounds + 1):
            batch = self.run_batch(specs=active_specs, decider=decider, stop=stop, iteration=iteration)
            results.append(batch)
            if not batch.dispatched or batch.ready_count == 0:
                break
        return tuple(results)


def _count(graph: TaskGraph, status: TaskStatus) -> int:
    return sum(1 for task in graph.tasks() if task.status == status)
