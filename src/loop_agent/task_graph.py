from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Tuple


class TaskStatus(str, Enum):
    pending = 'pending'
    ready = 'ready'
    running = 'running'
    blocked = 'blocked'
    completed = 'completed'
    failed = 'failed'


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    goal: str
    dependencies: Tuple[str, ...] = tuple()
    assignee: str | None = None
    status: TaskStatus = TaskStatus.pending
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'title': self.title,
            'goal': self.goal,
            'dependencies': list(self.dependencies),
            'assignee': self.assignee,
            'status': self.status.value,
            'metadata': self.metadata,
        }

    def to_store_dict(self, *, blocks: Tuple[str, ...] = tuple()) -> Dict[str, Any]:
        return {
            'id': self.id,
            'subject': self.title,
            'goal': self.goal,
            'status': self.status.value,
            'blockedBy': list(self.dependencies),
            'blocks': list(blocks),
            'assignee': self.assignee,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'Task':
        task_id = str(payload.get('id', '')).strip()
        title = str(payload.get('title', payload.get('subject', ''))).strip()
        goal = str(payload.get('goal', '')) if payload.get('goal') is not None else ''
        dependencies_raw = payload.get('dependencies', payload.get('blockedBy', []))
        dependencies = tuple(
            str(item).strip()
            for item in dependencies_raw
            if str(item).strip()
        ) if isinstance(dependencies_raw, (list, tuple)) else tuple()
        assignee_raw = payload.get('assignee')
        assignee = str(assignee_raw).strip() if isinstance(assignee_raw, str) and assignee_raw.strip() else None
        status_raw = str(payload.get('status', TaskStatus.pending.value)).strip() or TaskStatus.pending.value
        try:
            status = TaskStatus(status_raw)
        except ValueError:
            status = TaskStatus.pending
        metadata = payload.get('metadata', {})
        if not isinstance(metadata, dict):
            metadata = {}
        return cls(
            id=task_id,
            title=title or task_id,
            goal=goal or title or task_id,
            dependencies=dependencies,
            assignee=assignee,
            status=status,
            metadata=dict(metadata),
        )


class TaskGraph:
    def __init__(self, tasks: Iterable[Task] | None = None) -> None:
        self._tasks: Dict[str, Task] = {}
        if tasks is not None:
            for task in tasks:
                self.add_task(task)
        self.refresh_statuses()

    def add_task(self, task: Task) -> None:
        if not task.id.strip():
            raise ValueError('task id must not be empty')
        if task.id in self._tasks:
            raise ValueError(f'duplicate task id: {task.id}')
        self._tasks[task.id] = task
        self.validate()
        self.refresh_statuses()

    def get_task(self, task_id: str) -> Task:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise ValueError(f'unknown task id: {task_id}') from exc

    def tasks(self) -> Tuple[Task, ...]:
        return tuple(self._tasks.values())

    def ready_tasks(self) -> Tuple[Task, ...]:
        return tuple(task for task in self._tasks.values() if task.status == TaskStatus.ready)

    def validate(self) -> None:
        for task in self._tasks.values():
            for dependency in task.dependencies:
                if dependency not in self._tasks:
                    raise ValueError(f'unknown dependency: {dependency}')
        self._assert_acyclic()

    def set_status(self, task_id: str, status: TaskStatus, *, metadata: Dict[str, Any] | None = None) -> Task:
        task = self.get_task(task_id)
        next_metadata = dict(task.metadata)
        if metadata:
            next_metadata.update(metadata)
        updated = Task(
            id=task.id,
            title=task.title,
            goal=task.goal,
            dependencies=task.dependencies,
            assignee=task.assignee,
            status=status,
            metadata=next_metadata,
        )
        self._tasks[task_id] = updated
        self.refresh_statuses()
        return updated

    def assign_task(self, task_id: str, assignee: str) -> Task:
        task = self.get_task(task_id)
        updated = Task(
            id=task.id,
            title=task.title,
            goal=task.goal,
            dependencies=task.dependencies,
            assignee=assignee,
            status=task.status,
            metadata=dict(task.metadata),
        )
        self._tasks[task_id] = updated
        return updated

    def mark_running(self, task_id: str, *, metadata: Dict[str, Any] | None = None) -> Task:
        return self.set_status(task_id, TaskStatus.running, metadata=metadata)

    def mark_completed(self, task_id: str, *, metadata: Dict[str, Any] | None = None) -> Task:
        return self.set_status(task_id, TaskStatus.completed, metadata=metadata)

    def mark_failed(self, task_id: str, *, metadata: Dict[str, Any] | None = None) -> Task:
        return self.set_status(task_id, TaskStatus.failed, metadata=metadata)

    def refresh_statuses(self) -> None:
        updated: Dict[str, Task] = {}
        for task in self._tasks.values():
            if task.status in {TaskStatus.running, TaskStatus.completed, TaskStatus.failed}:
                updated[task.id] = task
                continue

            dependency_states = [self._tasks[dep].status for dep in task.dependencies if dep in self._tasks]
            if any(state == TaskStatus.failed for state in dependency_states):
                next_status = TaskStatus.blocked
            elif all(state == TaskStatus.completed for state in dependency_states):
                next_status = TaskStatus.ready
            else:
                next_status = TaskStatus.pending

            updated[task.id] = Task(
                id=task.id,
                title=task.title,
                goal=task.goal,
                dependencies=task.dependencies,
                assignee=task.assignee,
                status=next_status,
                metadata=dict(task.metadata),
            )
        self._tasks = updated

    def to_dict(self) -> Dict[str, Any]:
        return {'tasks': [task.to_dict() for task in self._tasks.values()]}

    def to_store_dict(self) -> Dict[str, Any]:
        reverse_edges = self.reverse_dependencies()
        return {
            'tasks': [
                task.to_store_dict(blocks=reverse_edges.get(task.id, tuple()))
                for task in self._tasks.values()
            ]
        }

    def reverse_dependencies(self) -> Dict[str, Tuple[str, ...]]:
        reverse: Dict[str, List[str]] = {task_id: [] for task_id in self._tasks}
        for task in self._tasks.values():
            for dependency in task.dependencies:
                reverse.setdefault(dependency, []).append(task.id)
        return {
            task_id: tuple(children)
            for task_id, children in reverse.items()
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'TaskGraph':
        tasks_raw = payload.get('tasks', [])
        if not isinstance(tasks_raw, list):
            raise ValueError('tasks must be a list')
        tasks = [Task.from_dict(item) for item in tasks_raw if isinstance(item, dict)]
        return cls(tasks)

    def _assert_acyclic(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visited:
                return
            if task_id in visiting:
                raise ValueError(f'cycle detected at task: {task_id}')
            visiting.add(task_id)
            for dependency in self._tasks[task_id].dependencies:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in self._tasks:
            visit(task_id)
