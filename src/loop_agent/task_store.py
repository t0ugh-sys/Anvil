from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

from .task_graph import Task, TaskGraph


@dataclass(frozen=True)
class TaskStore:
    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, '_lock', threading.RLock())

    def task_file(self, task_id: str) -> Path:
        return self.root_dir / f'task_{task_id}.json'

    def save_task(self, graph: TaskGraph, task_id: str) -> Path:
        with self._lock:
            reverse_edges = graph.reverse_dependencies()
            task = graph.get_task(task_id)
            path = self.task_file(task_id)
            payload = task.to_store_dict(blocks=reverse_edges.get(task_id, tuple()))
            self._write_json(path, payload)
            return path

    def save_graph(self, graph: TaskGraph) -> Tuple[Path, ...]:
        with self._lock:
            paths = []
            for task in graph.tasks():
                paths.append(self.save_task(graph, task.id))
            return tuple(paths)

    def load_graph(self) -> TaskGraph:
        with self._lock:
            tasks = []
            for path in sorted(self.root_dir.glob('task_*.json')):
                text = path.read_text(encoding='utf-8').strip()
                if not text:
                    continue
                payload = json.loads(text)
                if isinstance(payload, dict):
                    tasks.append(Task.from_dict(payload))
            return TaskGraph(tasks)

    def list_task_files(self) -> Tuple[Path, ...]:
        return tuple(sorted(self.root_dir.glob('task_*.json')))

    def replace_graph(self, tasks: Iterable[Task]) -> TaskGraph:
        graph = TaskGraph(tasks)
        self.save_graph(graph)
        return graph

    def _write_json(self, path: Path, payload: dict) -> None:
        temp_file = path.with_suffix(f'{path.suffix}.tmp')
        temp_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        temp_file.replace(path)
