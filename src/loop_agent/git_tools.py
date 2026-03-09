from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional

from .agent_protocol import ToolResult
from .tools import ToolContext


@dataclass(frozen=True)
class GitOptions:
    cwd: str
    timeout_s: float = 60.0


def _run_git(cmd: List[str], *, opts: GitOptions) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    return subprocess.run(
        cmd,
        cwd=opts.cwd,
        env=env,
        shell=False,
        check=False,
        text=True,
        capture_output=True,
        encoding='utf-8',
        errors='replace',
        timeout=opts.timeout_s,
    )


def _merge_output(proc: subprocess.CompletedProcess) -> str:
    return ((proc.stdout or '') + (proc.stderr or '')).strip()


def git_status_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    call_id = str(args.get('id', 'git_status'))
    opts = GitOptions(cwd=str(context.workspace_root), timeout_s=30.0)

    proc = _run_git(['git', 'status', '--porcelain=v1', '-b'], opts=opts)
    output = _merge_output(proc)
    ok = proc.returncode == 0
    return ToolResult(id=call_id, ok=ok, output=output, error=None if ok else 'failed to get git status')


def git_branch_list_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    call_id = str(args.get('id', 'git_branch_list'))
    opts = GitOptions(cwd=str(context.workspace_root), timeout_s=30.0)

    all_branches = bool(args.get('all', True))
    cmd = ['git', 'branch']
    if all_branches:
        cmd.append('-a')

    proc = _run_git(cmd, opts=opts)
    output = _merge_output(proc)
    ok = proc.returncode == 0
    return ToolResult(id=call_id, ok=ok, output=output, error=None if ok else 'failed to list branches')


def git_checkout_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    call_id = str(args.get('id', 'git_checkout'))
    opts = GitOptions(cwd=str(context.workspace_root), timeout_s=60.0)

    branch = str(args.get('branch', '')).strip()
    if not branch:
        return ToolResult(id=call_id, ok=False, output='', error='branch is required')

    proc = _run_git(['git', 'checkout', branch], opts=opts)
    output = _merge_output(proc)
    ok = proc.returncode == 0
    return ToolResult(id=call_id, ok=ok, output=output, error=None if ok else 'failed to checkout branch')


def git_pull_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    call_id = str(args.get('id', 'git_pull'))
    opts = GitOptions(cwd=str(context.workspace_root), timeout_s=120.0)

    remote = str(args.get('remote', 'origin')).strip() or 'origin'
    branch = str(args.get('branch', '')).strip()

    cmd = ['git', 'pull', '--ff-only', remote]
    if branch:
        cmd.append(branch)

    proc = _run_git(cmd, opts=opts)
    output = _merge_output(proc)
    ok = proc.returncode == 0
    return ToolResult(id=call_id, ok=ok, output=output, error=None if ok else 'failed to pull')


def git_merge_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    call_id = str(args.get('id', 'git_merge'))
    opts = GitOptions(cwd=str(context.workspace_root), timeout_s=120.0)

    source = str(args.get('source', '')).strip()
    target = str(args.get('target', '')).strip()
    strategy = str(args.get('strategy', 'merge')).strip().lower()
    confirm = bool(args.get('confirm', False))

    if not source:
        return ToolResult(id=call_id, ok=False, output='', error='source is required')
    if not target:
        return ToolResult(id=call_id, ok=False, output='', error='target is required')

    if strategy not in {'merge', 'ff-only', 'no-ff'}:
        strategy = 'merge'

    if not confirm:
        return ToolResult(
            id=call_id,
            ok=False,
            output='',
            error='refusing to merge without confirm=true',
        )

    co = _run_git(['git', 'checkout', target], opts=opts)
    co_out = _merge_output(co)
    if co.returncode != 0:
        return ToolResult(id=call_id, ok=False, output=co_out, error='failed to checkout target branch')

    cmd = ['git', 'merge']
    if strategy == 'ff-only':
        cmd.append('--ff-only')
    elif strategy == 'no-ff':
        cmd.append('--no-ff')
    cmd.append(source)

    proc = _run_git(cmd, opts=opts)
    output = (co_out + '\n' + _merge_output(proc)).strip() if co_out else _merge_output(proc)
    ok = proc.returncode == 0
    if not ok:
        return ToolResult(id=call_id, ok=False, output=output, error='merge failed (conflicts?)')

    return ToolResult(id=call_id, ok=True, output=output, error=None)


def git_push_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    call_id = str(args.get('id', 'git_push'))
    opts = GitOptions(cwd=str(context.workspace_root), timeout_s=120.0)

    remote = str(args.get('remote', 'origin')).strip() or 'origin'
    branch = str(args.get('branch', '')).strip()
    confirm = bool(args.get('confirm', False))

    if not branch:
        branch = 'HEAD'

    if not confirm:
        return ToolResult(id=call_id, ok=False, output='', error='refusing to push without confirm=true')

    proc = _run_git(['git', 'push', remote, branch], opts=opts)
    output = _merge_output(proc)
    ok = proc.returncode == 0
    return ToolResult(id=call_id, ok=ok, output=output, error=None if ok else 'failed to push')


def git_merge_and_push_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    call_id = str(args.get('id', 'git_merge_and_push'))
    opts = GitOptions(cwd=str(context.workspace_root), timeout_s=180.0)

    source = str(args.get('source', '')).strip()
    target = str(args.get('target', '')).strip()
    remote = str(args.get('remote', 'origin')).strip() or 'origin'
    strategy = str(args.get('strategy', 'merge')).strip().lower()
    pull = bool(args.get('pull', True))
    confirm = bool(args.get('confirm', False))

    if not source:
        return ToolResult(id=call_id, ok=False, output='', error='source is required')
    if not target:
        return ToolResult(id=call_id, ok=False, output='', error='target is required')
    if strategy not in {'merge', 'ff-only', 'no-ff'}:
        strategy = 'merge'

    if not confirm:
        return ToolResult(id=call_id, ok=False, output='', error='refusing to merge+push without confirm=true')

    # Ensure clean worktree
    st = _run_git(['git', 'status', '--porcelain=v1'], opts=GitOptions(cwd=opts.cwd, timeout_s=30.0))
    st_out = (st.stdout or '').strip()
    if st.returncode != 0:
        return ToolResult(id=call_id, ok=False, output=_merge_output(st), error='failed to get git status')
    if st_out:
        return ToolResult(id=call_id, ok=False, output=st_out, error='working tree not clean; commit/stash before merging')

    # Checkout target
    co = _run_git(['git', 'checkout', target], opts=GitOptions(cwd=opts.cwd, timeout_s=60.0))
    co_out = _merge_output(co)
    if co.returncode != 0:
        return ToolResult(id=call_id, ok=False, output=co_out, error='failed to checkout target branch')

    logs: List[str] = []
    if co_out:
        logs.append(co_out)

    # Pull latest (ff-only)
    if pull:
        pl = _run_git(['git', 'pull', '--ff-only', remote, target], opts=GitOptions(cwd=opts.cwd, timeout_s=120.0))
        pl_out = _merge_output(pl)
        if pl.returncode != 0:
            return ToolResult(id=call_id, ok=False, output='\n'.join(logs + [pl_out]).strip(), error='failed to pull (ff-only)')
        if pl_out:
            logs.append(pl_out)

    # Merge source
    mg_cmd = ['git', 'merge']
    if strategy == 'ff-only':
        mg_cmd.append('--ff-only')
    elif strategy == 'no-ff':
        mg_cmd.append('--no-ff')
    mg_cmd.append(source)

    mg = _run_git(mg_cmd, opts=GitOptions(cwd=opts.cwd, timeout_s=120.0))
    mg_out = _merge_output(mg)
    if mg.returncode != 0:
        return ToolResult(id=call_id, ok=False, output='\n'.join(logs + [mg_out]).strip(), error='merge failed (conflicts?)')
    if mg_out:
        logs.append(mg_out)

    # Push
    ps = _run_git(['git', 'push', remote, target], opts=GitOptions(cwd=opts.cwd, timeout_s=120.0))
    ps_out = _merge_output(ps)
    if ps.returncode != 0:
        return ToolResult(id=call_id, ok=False, output='\n'.join(logs + [ps_out]).strip(), error='push failed')
    if ps_out:
        logs.append(ps_out)

    return ToolResult(id=call_id, ok=True, output='\n'.join([l for l in logs if l]).strip(), error=None)
