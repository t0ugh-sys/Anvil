from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401

from loop_agent.llm.providers import _mock_invoke_factory
from loop_agent.mailbox import JsonlMailbox
from loop_agent.scheduler import TaskScheduler
from loop_agent.subagents import SubAgentRuntime, SubAgentSpec
from loop_agent.task_graph import Task, TaskGraph, TaskStatus


def _build_mock_decider():
    invoke = _mock_invoke_factory('mock-v3', mode='coding')

    def decider(goal, history, tool_results, state_summary, last_steps):
        payload = {
            'goal': goal,
            'history': list(history),
            'tool_results': [item.id for item in tool_results],
            'state_summary': state_summary,
            'last_steps': list(last_steps),
        }
        return invoke(json.dumps(payload, ensure_ascii=False))

    return decider


class SchedulerTests(unittest.TestCase):
    def test_should_run_ready_tasks_until_idle(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'scheduler-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / 'README.md').write_text('hello', encoding='utf-8')
        try:
            graph = TaskGraph(
                [
                    Task(id='t1', title='Inspect', goal='inspect README and finish'),
                    Task(id='t2', title='Summarize', goal='summarize findings', dependencies=('t1',)),
                ]
            )
            mailbox = JsonlMailbox(tmp_dir / 'mailbox')
            runtime = SubAgentRuntime(mailbox=mailbox, task_graph=graph)
            scheduler = TaskScheduler(runtime=runtime, max_parallel_agents=1)
            spec = SubAgentSpec(agent_id='worker-1', role='reader', workspace_root=tmp_dir)
            batches = scheduler.run_until_idle(specs=(spec,), decider=_build_mock_decider())
            self.assertGreaterEqual(len(batches), 2)
            self.assertEqual(graph.get_task('t1').status, TaskStatus.completed)
            self.assertEqual(graph.get_task('t2').status, TaskStatus.completed)
            self.assertEqual(batches[-1].ready_count, 0)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_limit_parallel_agents_per_batch(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'scheduler-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / 'README.md').write_text('hello', encoding='utf-8')
        try:
            graph = TaskGraph(
                [
                    Task(id='t1', title='Inspect A', goal='inspect README and finish'),
                    Task(id='t2', title='Inspect B', goal='inspect README and finish'),
                ]
            )
            mailbox = JsonlMailbox(tmp_dir / 'mailbox')
            runtime = SubAgentRuntime(mailbox=mailbox, task_graph=graph)
            scheduler = TaskScheduler(runtime=runtime, max_parallel_agents=1)
            specs = (
                SubAgentSpec(agent_id='worker-1', role='reader', workspace_root=tmp_dir),
                SubAgentSpec(agent_id='worker-2', role='reader', workspace_root=tmp_dir),
            )
            batch = scheduler.run_batch(specs=specs, decider=_build_mock_decider())
            self.assertEqual(len(batch.dispatched), 1)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
