from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401

from loop_agent.ops.git_tools import git_merge_and_push_tool
from loop_agent.tools import ToolContext


class _Proc:
    def __init__(self, *, returncode: int = 0, stdout: str = '', stderr: str = ''):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class GitToolsTests(unittest.TestCase):
    def test_merge_and_push_requires_confirm(self) -> None:
        ctx = ToolContext(workspace_root=Path('.'))

        def fake_run(*args, **kwargs):
            raise AssertionError('should short-circuit before running git')

        with patch.object(subprocess, 'run', side_effect=fake_run):
            res = git_merge_and_push_tool(
                ctx,
                {'source': 'feature', 'target': 'main', 'confirm': False, 'id': 'x'},
            )

        self.assertFalse(res.ok)
        self.assertIn('confirm=true', res.error or '')

    def test_merge_and_push_refuses_dirty_tree(self) -> None:
        ctx = ToolContext(workspace_root=Path('.'))

        def fake_run(cmd, cwd, env, shell, check, text, capture_output, encoding, errors, timeout):
            if cmd[:2] == ['git', 'status']:
                return _Proc(returncode=0, stdout=' M file.txt\n')
            raise AssertionError(f'unexpected cmd: {cmd}')

        with patch.object(subprocess, 'run', side_effect=fake_run):
            res = git_merge_and_push_tool(
                ctx,
                {'source': 'feature', 'target': 'main', 'confirm': True, 'id': 'x'},
            )

        self.assertFalse(res.ok)
        self.assertIn('not clean', res.error or '')


if __name__ == '__main__':
    unittest.main()
