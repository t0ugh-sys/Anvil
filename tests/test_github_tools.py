from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401

from loop_agent.ops.github_tools import (
    _parse_repo_from_remote,
    _resolve_repo_arg,
    gh_issue_close_tool,
    gh_issue_list_tool,
    gh_pr_merge_tool,
)
from loop_agent.tools import ToolContext


class _Proc:
    def __init__(self, *, returncode: int = 0, stdout: str = '', stderr: str = ''):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class GitHubToolsTests(unittest.TestCase):
    def test_parse_repo_from_remote(self) -> None:
        self.assertEqual(_parse_repo_from_remote('git@github.com:foo/bar.git'), 'foo/bar')
        self.assertEqual(_parse_repo_from_remote('git@github.com:foo/bar'), 'foo/bar')
        self.assertEqual(_parse_repo_from_remote('https://github.com/foo/bar.git'), 'foo/bar')
        self.assertEqual(_parse_repo_from_remote('https://github.com/foo/bar'), 'foo/bar')
        self.assertIsNone(_parse_repo_from_remote(''))
        self.assertIsNone(_parse_repo_from_remote('git@gitlab.com:foo/bar.git'))

    def test_resolve_repo_arg_prefers_direct_arg(self) -> None:
        ctx = ToolContext(workspace_root=Path('.'))

        def fake_run(*args, **kwargs):
            raise AssertionError('should not call git when repo is provided')

        with patch.object(subprocess, 'run', side_effect=fake_run):
            repo, err = _resolve_repo_arg(ctx, 'foo/bar')

        self.assertEqual(repo, 'foo/bar')
        self.assertIsNone(err)

    def test_resolve_repo_arg_from_origin_remote(self) -> None:
        ctx = ToolContext(workspace_root=Path('.'))

        def fake_run(cmd, cwd, env, shell, check, text, capture_output, encoding, errors, timeout):
            self.assertEqual(cmd[:2], ['git', 'remote'])
            return _Proc(
                returncode=0,
                stdout=(
                    'origin  git@github.com:hello/world.git (fetch)\n'
                    'origin  git@github.com:hello/world.git (push)\n'
                ),
            )

        with patch.object(subprocess, 'run', side_effect=fake_run):
            repo, err = _resolve_repo_arg(ctx, '')

        self.assertEqual(repo, 'hello/world')
        self.assertIsNone(err)

    def test_issue_close_requires_confirm(self) -> None:
        ctx = ToolContext(workspace_root=Path('.'))

        def fake_run(cmd, cwd, env, shell, check, text, capture_output, encoding, errors, timeout):
            if cmd[:2] == ['gh', '--version']:
                return _Proc(returncode=0, stdout='gh version 2.x')
            if cmd[:2] == ['git', 'remote']:
                return _Proc(returncode=1, stdout='', stderr='no git')
            raise AssertionError(f'unexpected cmd: {cmd}')

        with patch.object(subprocess, 'run', side_effect=fake_run):
            res = gh_issue_close_tool(ctx, {'repo': 'foo/bar', 'number': 1, 'confirm': False, 'id': 'x'})

        self.assertFalse(res.ok)
        self.assertIn('confirm=true', res.error or '')

    def test_pr_merge_requires_confirm(self) -> None:
        ctx = ToolContext(workspace_root=Path('.'))

        def fake_run(cmd, cwd, env, shell, check, text, capture_output, encoding, errors, timeout):
            if cmd[:2] == ['gh', '--version']:
                return _Proc(returncode=0, stdout='gh version 2.x')
            if cmd[:2] == ['git', 'remote']:
                return _Proc(returncode=1, stdout='', stderr='no git')
            raise AssertionError(f'unexpected cmd: {cmd}')

        with patch.object(subprocess, 'run', side_effect=fake_run):
            res = gh_pr_merge_tool(ctx, {'repo': 'foo/bar', 'number': 2, 'confirm': False, 'id': 'x'})

        self.assertFalse(res.ok)
        self.assertIn('confirm=true', res.error or '')

    def test_issue_list_can_infer_repo(self) -> None:
        ctx = ToolContext(workspace_root=Path('.'))
        calls = []

        def fake_run(cmd, cwd, env, shell, check, text, capture_output, encoding, errors, timeout):
            calls.append(cmd)
            if cmd[:2] == ['gh', '--version']:
                return _Proc(returncode=0, stdout='gh version 2.x')
            if cmd[:2] == ['git', 'remote']:
                return _Proc(
                    returncode=0,
                    stdout='origin  https://github.com/aa/bb.git (fetch)\n',
                )
            if cmd[:3] == ['gh', 'issue', 'list']:
                self.assertIn('--repo', cmd)
                idx = cmd.index('--repo')
                self.assertEqual(cmd[idx + 1], 'aa/bb')
                return _Proc(returncode=0, stdout='[]')
            raise AssertionError(f'unexpected cmd: {cmd}')

        with patch.object(subprocess, 'run', side_effect=fake_run):
            res = gh_issue_list_tool(ctx, {'repo': '', 'state': 'open', 'limit': 5, 'id': 'x'})

        self.assertTrue(res.ok)
        self.assertEqual(res.output, '')
        self.assertTrue(any(c[:2] == ['git', 'remote'] for c in calls))


if __name__ == '__main__':
    unittest.main()
