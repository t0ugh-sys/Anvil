from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from loop_agent.task_graph import Task, TaskGraph, TaskStatus


class TaskGraphTests(unittest.TestCase):
    def test_should_mark_root_tasks_ready(self) -> None:
        graph = TaskGraph([Task(id='t1', title='T1', goal='g1')])
        self.assertEqual(graph.get_task('t1').status, TaskStatus.ready)

    def test_should_unlock_dependent_task_after_completion(self) -> None:
        graph = TaskGraph(
            [
                Task(id='t1', title='T1', goal='g1'),
                Task(id='t2', title='T2', goal='g2', dependencies=('t1',)),
            ]
        )
        self.assertEqual(graph.get_task('t2').status, TaskStatus.pending)
        graph.mark_completed('t1')
        self.assertEqual(graph.get_task('t2').status, TaskStatus.ready)

    def test_should_reject_cycles(self) -> None:
        with self.assertRaises(ValueError):
            TaskGraph(
                [
                    Task(id='t1', title='T1', goal='g1', dependencies=('t2',)),
                    Task(id='t2', title='T2', goal='g2', dependencies=('t1',)),
                ]
            )


if __name__ == '__main__':
    unittest.main()
