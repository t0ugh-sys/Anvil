from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401

from loop_agent.agent_protocol import ToolCall
from loop_agent.permissions import PermissionManager
from loop_agent.policies import Capability, ToolPolicy
from loop_agent.tools import ToolContext, build_default_tools, execute_tool_call


class PermissionManagerTests(unittest.TestCase):
    def test_should_deny_write_in_strict_mode(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'perm-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            manager = PermissionManager(mode_name='strict')
            policy = ToolPolicy(allowed=tuple(Capability), permission_manager=manager)
            call = ToolCall(id='call_1', name='write_file', arguments={'path': 'x.txt', 'content': 'x'})
            result = execute_tool_call(ToolContext(workspace_root=tmp_dir, policy=policy), call, build_default_tools())
            self.assertFalse(result.ok)
            self.assertEqual(result.metadata.get('permission_decision'), 'deny')
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_require_approval_for_write_in_balanced_mode(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'perm-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            manager = PermissionManager(mode_name='balanced')
            policy = ToolPolicy(allowed=tuple(Capability), permission_manager=manager)
            call = ToolCall(id='call_1', name='write_file', arguments={'path': 'x.txt', 'content': 'x'})
            result = execute_tool_call(ToolContext(workspace_root=tmp_dir, policy=policy), call, build_default_tools())
            self.assertFalse(result.ok)
            self.assertEqual(result.metadata.get('permission_decision'), 'ask')
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_allow_reads_in_balanced_mode(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'perm-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / 'README.md').write_text('hello', encoding='utf-8')
        try:
            manager = PermissionManager(mode_name='balanced')
            policy = ToolPolicy(allowed=tuple(Capability), permission_manager=manager)
            call = ToolCall(id='call_1', name='read_file', arguments={'path': 'README.md'})
            result = execute_tool_call(ToolContext(workspace_root=tmp_dir, policy=policy), call, build_default_tools())
            self.assertTrue(result.ok)
            self.assertEqual(result.metadata.get('permission_decision'), 'allow')
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_allow_all_tools_in_unsafe_mode(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'perm-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / 'README.md').write_text('hello', encoding='utf-8')
        try:
            manager = PermissionManager(mode_name='unsafe')
            policy = ToolPolicy(allowed=tuple(Capability), permission_manager=manager)
            call = ToolCall(id='call_1', name='read_file', arguments={'path': 'README.md'})
            result = execute_tool_call(ToolContext(workspace_root=tmp_dir, policy=policy), call, build_default_tools())
            self.assertTrue(result.ok)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_reuse_cached_permission_decision(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'perm-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            manager = PermissionManager(mode_name='balanced', cache={'write_file:write': 'deny'})
            policy = ToolPolicy(allowed=tuple(Capability), permission_manager=manager)
            call = ToolCall(id='call_1', name='write_file', arguments={'path': 'x.txt', 'content': 'x'})
            result = execute_tool_call(ToolContext(workspace_root=tmp_dir, policy=policy), call, build_default_tools())
            self.assertFalse(result.ok)
            self.assertTrue(result.metadata.get('permission_cached'))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
