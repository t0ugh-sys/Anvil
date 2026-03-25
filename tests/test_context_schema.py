from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401

from loop_agent.context_schema import OrchestrationContextInput, build_orchestration_context
from loop_agent.mailbox import JsonlMailbox, MailMessage
from loop_agent.policies import ToolPolicy
from loop_agent.task_graph import Task, TaskGraph


class ContextSchemaTests(unittest.TestCase):
    def test_should_build_fixed_orchestration_schema(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'context-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            mailbox = JsonlMailbox(tmp_dir / 'mailbox')
            mailbox.send(
                MailMessage(
                    id='m1',
                    sender='coordinator',
                    recipient='worker-1',
                    subject='Assigned task t1',
                    body='inspect README',
                    task_id='t1',
                )
            )
            graph = TaskGraph([Task(id='t1', title='Inspect', goal='inspect README and finish')])
            context = build_orchestration_context(
                OrchestrationContextInput(
                    goal='inspect README and finish',
                    agent_id='worker-1',
                    current_task_id='t1',
                    workspace_root=tmp_dir,
                    mailbox=mailbox,
                    task_graph=graph,
                    policy=ToolPolicy.read_only(),
                    facts=('prefer local files first',),
                    current_plan=('read README', 'summarize findings'),
                    recent_steps=('step-1',),
                    isolation_mode='copy',
                )
            )
            self.assertEqual(context['context_schema'], 'orchestration-v1')
            self.assertEqual(context['agent']['agent_id'], 'worker-1')
            self.assertEqual(context['task_state']['current_task_id'], 't1')
            self.assertIn('t1', context['task_state']['ready'])
            self.assertEqual(context['mailbox_digest']['count'], 1)
            self.assertEqual(context['policy']['allowed'], ['read', 'memory'])
            self.assertEqual(context['workspace']['isolation_mode'], 'copy')
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
