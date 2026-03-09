from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from loop_agent.git_tools import git_merge_and_push_tool
from loop_agent.tools import ToolContext


class _Proc:
    def __init__(self, *, returncode: int = 0, stdout: str = '', stderr: str = ''):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_merge_and_push_requires_confirm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = ToolContext(workspace_root=tmp_path)

    def fake_run(*args, **kwargs):
        raise AssertionError('should short-circuit before running git')

    monkeypatch.setattr(subprocess, 'run', fake_run)

    res = git_merge_and_push_tool(
        ctx,
        {'source': 'feature', 'target': 'main', 'confirm': False, 'id': 'x'},
    )
    assert res.ok is False
    assert 'confirm=true' in (res.error or '')


def test_merge_and_push_refuses_dirty_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = ToolContext(workspace_root=tmp_path)

    def fake_run(cmd, cwd, env, shell, check, text, capture_output, encoding, errors, timeout):
        if cmd[:2] == ['git', 'status']:
            return _Proc(returncode=0, stdout=' M file.txt\n')
        raise AssertionError(f'unexpected cmd: {cmd}')

    monkeypatch.setattr(subprocess, 'run', fake_run)

    res = git_merge_and_push_tool(
        ctx,
        {'source': 'feature', 'target': 'main', 'confirm': True, 'id': 'x'},
    )
    assert res.ok is False
    assert 'not clean' in (res.error or '')
