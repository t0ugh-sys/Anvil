from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401

from loop_agent.agent_protocol import ToolCall
from loop_agent.policies import Capability, ToolPolicy, policy_from_name
from loop_agent.tools import ToolContext, build_default_tools, execute_tool_call


class ToolPolicyTests(unittest.TestCase):
    def test_should_block_write_tool_in_read_only_policy(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'policy-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            call = ToolCall(id='call_1', name='write_file', arguments={'path': 'x.txt', 'content': 'x'})
            context = ToolContext(workspace_root=tmp_dir, policy=ToolPolicy.read_only())
            result = execute_tool_call(context, call, build_default_tools())
            self.assertFalse(result.ok)
            self.assertIn('tool blocked by policy', result.error or '')
            self.assertFalse((tmp_dir / 'x.txt').exists())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_allow_read_tool_in_read_only_policy(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'policy-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        target = tmp_dir / 'README.md'
        target.write_text('hello', encoding='utf-8')
        try:
            call = ToolCall(id='call_1', name='read_file', arguments={'path': 'README.md'})
            context = ToolContext(workspace_root=tmp_dir, policy=ToolPolicy.read_only())
            result = execute_tool_call(context, call, build_default_tools())
            self.assertTrue(result.ok)
            self.assertEqual(result.output, 'hello')
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_parse_policy_preset(self) -> None:
        self.assertEqual(policy_from_name('read_only').allowed, (Capability.read, Capability.memory))


if __name__ == '__main__':
    unittest.main()
