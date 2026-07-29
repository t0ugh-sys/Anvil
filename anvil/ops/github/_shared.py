from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ...tools import ToolContext

__all__: List[str] = []


@dataclass(frozen=True)
class GhOptions:
    cwd: str
    timeout_s: float = 60.0


def _run(cmd: List[str], *, cwd: str, timeout_s: float) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        shell=False,
        check=False,
        text=True,
        capture_output=True,
        encoding='utf-8',
        errors='replace',
        timeout=timeout_s,
    )


def _run_gh(cmd: List[str], *, opts: GhOptions) -> subprocess.CompletedProcess:
    return _run(cmd, cwd=opts.cwd, timeout_s=opts.timeout_s)


def _merge_output(proc: subprocess.CompletedProcess) -> str:
    return ((proc.stdout or '') + (proc.stderr or '')).strip()


def _require_gh_available(opts: GhOptions) -> Optional[str]:
    proc = _run_gh(['gh', '--version'], opts=opts)
    if proc.returncode != 0:
        return 'gh CLI not available. Install from https://cli.github.com/ and ensure `gh` is on PATH.'
    return None


def _strip_suffix(value: str, suffix: str) -> str:
    if value.endswith(suffix):
        return value[: -len(suffix)]
    return value


def _normalize_repo(repo: str) -> str:
    return _strip_suffix(repo.strip(), '.git')


def _parse_repo_from_remote(url: str) -> Optional[str]:
    u = url.strip()
    if not u:
        return None

    # SSH: git@github.com:owner/name.git
    if u.startswith('git@github.com:'):
        rest = u[len('git@github.com:') :]
        rest = _strip_suffix(rest.strip(), '.git')
        if '/' in rest:
            return rest
        return None

    # HTTPS: https://github.com/owner/name.git
    if u.startswith('https://github.com/'):
        rest = u[len('https://github.com/') :]
        rest = _strip_suffix(rest.strip(), '.git')
        parts = [p for p in rest.split('/') if p]
        if len(parts) >= 2:
            return f'{parts[0]}/{parts[1]}'
        return None

    return None


def _resolve_repo_arg(context: ToolContext, repo_arg: str) -> Tuple[Optional[str], Optional[str]]:
    repo = _normalize_repo(repo_arg)
    if repo:
        return repo, None

    # Try infer from current workspace git remote.
    # Prefer origin, fallback to first remote.
    proc = _run(['git', 'remote', '-v'], cwd=str(context.workspace_root), timeout_s=10.0)
    if proc.returncode != 0:
        return None, 'repo is required (owner/name). Could not infer because git remote is unavailable.'

    remotes = []
    for line in (proc.stdout or '').splitlines():
        # origin  git@github.com:owner/name.git (fetch)
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0]
        url = parts[1]
        remotes.append((name, url))

    # Try origin first
    for name, url in remotes:
        if name == 'origin':
            parsed = _parse_repo_from_remote(url)
            if parsed:
                return parsed, None

    for _, url in remotes:
        parsed = _parse_repo_from_remote(url)
        if parsed:
            return parsed, None

    return None, 'repo is required (owner/name). Could not infer from git remotes.'
