from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401

from loop_agent.task_graph import Task, TaskGraph, TaskStatus
from loop_agent.task_store import TaskStore


class TaskStoreTests(unittest.TestCase):
    def test_should_persist_tasks_as_individual_json_files(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'task-store-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            store = TaskStore(tmp_dir / '.tasks')
            graph = TaskGraph(
                [
                    Task(id='1', title='Task 1', goal='g1'),
                    Task(id='2', title='Task 2', goal='g2', dependencies=('1',)),
                ]
            )

            paths = store.save_graph(graph)

            self.assertEqual(len(paths), 2)
            payload = json.loads((tmp_dir / '.tasks' / 'task_2.json').read_text(encoding='utf-8'))
            self.assertEqual(payload['blockedBy'], ['1'])
            self.assertEqual(payload['blocks'], [])
            self.assertEqual(payload['status'], 'pending')
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_unlock_blocked_task_after_reload(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'task-store-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            store = TaskStore(tmp_dir / '.tasks')
            graph = TaskGraph(
                [
                    Task(id='1', title='Task 1', goal='g1'),
                    Task(id='2', title='Task 2', goal='g2', dependencies=('1',)),
                    Task(id='3', title='Task 3', goal='g3', dependencies=('2',)),
                ]
            )
            store.save_graph(graph)

            graph.mark_completed('1')
            store.save_graph(graph)
            reloaded = store.load_graph()

            self.assertEqual(reloaded.get_task('2').status, TaskStatus.ready)
            self.assertEqual(reloaded.get_task('3').status, TaskStatus.pending)
            payload = json.loads((tmp_dir / '.tasks' / 'task_1.json').read_text(encoding='utf-8'))
            self.assertEqual(payload['blocks'], ['2'])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
