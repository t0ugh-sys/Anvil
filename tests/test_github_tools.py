from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from loop_agent.github_tools import (
    GhOptions,
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


def test_parse_repo_from_remote() -> None:
    assert _parse_repo_from_remote('git@github.com:foo/bar.git') == 'foo/bar'
    assert _parse_repo_from_remote('git@github.com:foo/bar') == 'foo/bar'
    assert _parse_repo_from_remote('https://github.com/foo/bar.git') == 'foo/bar'
    assert _parse_repo_from_remote('https://github.com/foo/bar') == 'foo/bar'
    assert _parse_repo_from_remote('') is None
    assert _parse_repo_from_remote('git@gitlab.com:foo/bar.git') is None


def test_resolve_repo_arg_prefers_direct_arg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = ToolContext(workspace_root=tmp_path)

    def fake_run(*args, **kwargs):
        raise AssertionError('should not call git when repo is provided')

    monkeypatch.setattr(subprocess, 'run', fake_run)

    repo, err = _resolve_repo_arg(ctx, 'foo/bar')
    assert repo == 'foo/bar'
    assert err is None


def test_resolve_repo_arg_from_origin_remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = ToolContext(workspace_root=tmp_path)

    def fake_run(cmd, cwd, env, shell, check, text, capture_output, encoding, errors, timeout):
        assert cmd[:2] == ['git', 'remote']
        return _Proc(
            returncode=0,
            stdout=(
                'origin  git@github.com:hello/world.git (fetch)\n'
                'origin  git@github.com:hello/world.git (push)\n'
            ),
        )

    monkeypatch.setattr(subprocess, 'run', fake_run)

    repo, err = _resolve_repo_arg(ctx, '')
    assert repo == 'hello/world'
    assert err is None


def test_issue_close_requires_confirm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = ToolContext(workspace_root=tmp_path)

    def fake_run(cmd, cwd, env, shell, check, text, capture_output, encoding, errors, timeout):
        # gh --version
        if cmd[:2] == ['gh', '--version']:
            return _Proc(returncode=0, stdout='gh version 2.x')
        # git remote -v, but for this test we always pass repo
        if cmd[:2] == ['git', 'remote']:
            return _Proc(returncode=1, stdout='', stderr='no git')
        raise AssertionError(f'unexpected cmd: {cmd}')

    monkeypatch.setattr(subprocess, 'run', fake_run)

    res = gh_issue_close_tool(ctx, {'repo': 'foo/bar', 'number': 1, 'confirm': False, 'id': 'x'})
    assert res.ok is False
    assert 'confirm=true' in (res.error or '')


def test_pr_merge_requires_confirm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = ToolContext(workspace_root=tmp_path)

    def fake_run(cmd, cwd, env, shell, check, text, capture_output, encoding, errors, timeout):
        if cmd[:2] == ['gh', '--version']:
            return _Proc(returncode=0, stdout='gh version 2.x')
        if cmd[:2] == ['git', 'remote']:
            return _Proc(returncode=1, stdout='', stderr='no git')
        raise AssertionError(f'unexpected cmd: {cmd}')

    monkeypatch.setattr(subprocess, 'run', fake_run)

    res = gh_pr_merge_tool(ctx, {'repo': 'foo/bar', 'number': 2, 'confirm': False, 'id': 'x'})
    assert res.ok is False
    assert 'confirm=true' in (res.error or '')


def test_issue_list_can_infer_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = ToolContext(workspace_root=tmp_path)

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
            assert '--repo' in cmd
            idx = cmd.index('--repo')
            assert cmd[idx + 1] == 'aa/bb'
            return _Proc(returncode=0, stdout='[]')
        raise AssertionError(f'unexpected cmd: {cmd}')

    monkeypatch.setattr(subprocess, 'run', fake_run)

    res = gh_issue_list_tool(ctx, {'repo': '', 'state': 'open', 'limit': 5, 'id': 'x'})
    assert res.ok is True
    assert res.output == ''
    assert any(c[:2] == ['git', 'remote'] for c in calls)
