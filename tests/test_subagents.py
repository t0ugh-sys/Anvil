from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401

from loop_agent.llm.providers import _mock_invoke_factory
from loop_agent.mailbox import JsonlMailbox
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


class SubAgentRuntimeTests(unittest.TestCase):
    def test_should_dispatch_ready_task_and_write_mail(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'subagents-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / 'README.md').write_text('hello', encoding='utf-8')
        try:
            graph = TaskGraph([Task(id='t1', title='Inspect', goal='inspect README and finish')])
            mailbox = JsonlMailbox(tmp_dir / 'mailbox')
            runtime = SubAgentRuntime(mailbox=mailbox, task_graph=graph)
            spec = SubAgentSpec(agent_id='worker-1', role='reader', workspace_root=tmp_dir)
            results = runtime.dispatch_ready_tasks(specs=(spec,), decider=_build_mock_decider())
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].success)
            self.assertEqual(graph.get_task('t1').status, TaskStatus.completed)
            inbox = mailbox.inbox('coordinator')
            self.assertTrue(any(item.subject == 'Finished task t1' for item in inbox))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_keep_dependent_task_pending_until_unblocked(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'subagents-{uuid.uuid4().hex}'
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
            spec = SubAgentSpec(agent_id='worker-1', role='reader', workspace_root=tmp_dir)
            first = runtime.dispatch_ready_tasks(specs=(spec,), decider=_build_mock_decider())
            second = runtime.dispatch_ready_tasks(specs=(spec,), decider=_build_mock_decider())
            self.assertEqual(len(first), 1)
            self.assertEqual(len(second), 1)
            self.assertEqual(graph.get_task('t2').status, TaskStatus.completed)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
