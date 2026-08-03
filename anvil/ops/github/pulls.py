from __future__ import annotations

import json
from typing import Dict, List

from ...agent.protocol import ToolResult
from ...tools import ToolContext
from ._shared import GhOptions, _merge_output, _require_gh_available, _resolve_repo_arg, _run_gh

__all__ = [
    'gh_pr_list_tool',
    'gh_pr_create_tool',
    'gh_pr_view_tool',
    'gh_pr_checks_tool',
    'gh_pr_comment_tool',
    'gh_pr_merge_tool',
]


def gh_pr_list_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    call_id = str(args.get('id', 'gh_pr_list'))
    opts = GhOptions(cwd=str(context.workspace_root))

    repo_arg = str(args.get('repo', '')).strip()
    repo, err = _resolve_repo_arg(context, repo_arg)
    if err:
        return ToolResult(id=call_id, ok=False, output='', error=err)

    state = str(args.get('state', 'open')).strip().lower()
    limit = int(str(args.get('limit', '20')))
    if state not in {'open', 'closed', 'merged', 'all'}:
        state = 'open'
    if limit <= 0:
        limit = 20

    missing = _require_gh_available(opts)
    if missing:
        return ToolResult(id=call_id, ok=False, output='', error=missing)

    proc = _run_gh(
        [
            'gh',
            'pr',
            'list',
            '--repo',
            str(repo),
            '--state',
            state,
            '--limit',
            str(limit),
            '--json',
            'number,title,url,state,author,headRefName,baseRefName',
        ],
        opts=opts,
    )
    output = _merge_output(proc)
    if proc.returncode != 0:
        return ToolResult(id=call_id, ok=False, output=output, error='failed to list PRs')

    try:
        data = json.loads(proc.stdout or '[]')
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output=proc.stdout or '', error=f'JSON parse error: {exc}')

    lines: List[str] = []
    for item in data:
        number = item.get('number')
        title = str(item.get('title') or '')
        url = str(item.get('url') or '')
        st = str(item.get('state') or '')
        head = str(item.get('headRefName') or '')
        base = str(item.get('baseRefName') or '')
        author = ''
        try:
            author = str((item.get('author') or {}).get('login') or '')
        except Exception:
            author = ''
        lines.append(f'#{number} {title} [{st}] {head}->{base} ({author}) {url}'.strip())

    return ToolResult(id=call_id, ok=True, output='\n'.join(lines), error=None)


def gh_pr_create_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    call_id = str(args.get('id', 'gh_pr_create'))
    opts = GhOptions(cwd=str(context.workspace_root))

    repo_arg = str(args.get('repo', '')).strip()
    repo, err = _resolve_repo_arg(context, repo_arg)
    if err:
        return ToolResult(id=call_id, ok=False, output='', error=err)

    title = str(args.get('title', '')).strip()
    body = str(args.get('body', '')).strip()
    base = str(args.get('base', '')).strip()
    head = str(args.get('head', '')).strip()
    draft = bool(args.get('draft', False))

    if not title:
        return ToolResult(id=call_id, ok=False, output='', error='title is required')

    missing = _require_gh_available(opts)
    if missing:
        return ToolResult(id=call_id, ok=False, output='', error=missing)

    cmd: List[str] = ['gh', 'pr', 'create', '--repo', str(repo), '--title', title]
    if body:
        cmd.extend(['--body', body])
    if base:
        cmd.extend(['--base', base])
    if head:
        cmd.extend(['--head', head])
    if draft:
        cmd.append('--draft')

    proc = _run_gh(cmd, opts=opts)
    output = _merge_output(proc)
    ok = proc.returncode == 0
    return ToolResult(id=call_id, ok=ok, output=output, error=None if ok else 'failed to create PR')


def gh_pr_view_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    call_id = str(args.get('id', 'gh_pr_view'))
    opts = GhOptions(cwd=str(context.workspace_root))

    repo_arg = str(args.get('repo', '')).strip()
    repo, err = _resolve_repo_arg(context, repo_arg)
    if err:
        return ToolResult(id=call_id, ok=False, output='', error=err)

    number = str(args.get('number', '')).strip()
    if not number:
        return ToolResult(id=call_id, ok=False, output='', error='number is required')

    missing = _require_gh_available(opts)
    if missing:
        return ToolResult(id=call_id, ok=False, output='', error=missing)

    proc = _run_gh(
        [
            'gh',
            'pr',
            'view',
            number,
            '--repo',
            str(repo),
            '--json',
            'number,title,url,state,isDraft,mergeable,headRefName,baseRefName,author,reviewDecision,statusCheckRollup',
        ],
        opts=opts,
    )
    output = _merge_output(proc)
    ok = proc.returncode == 0
    return ToolResult(id=call_id, ok=ok, output=output, error=None if ok else 'failed to view PR')


def gh_pr_checks_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    call_id = str(args.get('id', 'gh_pr_checks'))
    opts = GhOptions(cwd=str(context.workspace_root))

    repo_arg = str(args.get('repo', '')).strip()
    repo, err = _resolve_repo_arg(context, repo_arg)
    if err:
        return ToolResult(id=call_id, ok=False, output='', error=err)

    number = str(args.get('number', '')).strip()
    if not number:
        return ToolResult(id=call_id, ok=False, output='', error='number is required')

    missing = _require_gh_available(opts)
    if missing:
        return ToolResult(id=call_id, ok=False, output='', error=missing)

    proc = _run_gh(['gh', 'pr', 'checks', number, '--repo', str(repo)], opts=opts)
    output = _merge_output(proc)
    ok = proc.returncode == 0
    return ToolResult(id=call_id, ok=ok, output=output, error=None if ok else 'failed to get PR checks')


def gh_pr_comment_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    call_id = str(args.get('id', 'gh_pr_comment'))
    opts = GhOptions(cwd=str(context.workspace_root))

    repo_arg = str(args.get('repo', '')).strip()
    repo, err = _resolve_repo_arg(context, repo_arg)
    if err:
        return ToolResult(id=call_id, ok=False, output='', error=err)

    number = str(args.get('number', '')).strip()
    body = str(args.get('body', '')).strip()

    if not number:
        return ToolResult(id=call_id, ok=False, output='', error='number is required')
    if not body:
        return ToolResult(id=call_id, ok=False, output='', error='body is required')

    missing = _require_gh_available(opts)
    if missing:
        return ToolResult(id=call_id, ok=False, output='', error=missing)

    proc = _run_gh(['gh', 'pr', 'comment', number, '--repo', str(repo), '--body', body], opts=opts)
    output = _merge_output(proc)
    ok = proc.returncode == 0
    return ToolResult(id=call_id, ok=ok, output=output, error=None if ok else 'failed to comment on PR')


def gh_pr_merge_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    call_id = str(args.get('id', 'gh_pr_merge'))
    opts = GhOptions(cwd=str(context.workspace_root))

    repo_arg = str(args.get('repo', '')).strip()
    repo, err = _resolve_repo_arg(context, repo_arg)
    if err:
        return ToolResult(id=call_id, ok=False, output='', error=err)

    number = str(args.get('number', '')).strip()
    method = str(args.get('method', 'merge')).strip().lower()
    delete_branch = bool(args.get('delete_branch', True))
    confirm = bool(args.get('confirm', False))

    if not number:
        return ToolResult(id=call_id, ok=False, output='', error='number is required')
    if method not in {'merge', 'squash', 'rebase'}:
        method = 'merge'
    if not confirm:
        return ToolResult(id=call_id, ok=False, output='', error='refusing to merge PR without confirm=true')

    missing = _require_gh_available(opts)
    if missing:
        return ToolResult(id=call_id, ok=False, output='', error=missing)

    cmd: List[str] = ['gh', 'pr', 'merge', number, '--repo', str(repo)]
    if method == 'merge':
        cmd.append('--merge')
    elif method == 'squash':
        cmd.append('--squash')
    else:
        cmd.append('--rebase')
    if delete_branch:
        cmd.append('--delete-branch')
    cmd.append('--yes')

    proc = _run_gh(cmd, opts=opts)
    output = _merge_output(proc)
    ok = proc.returncode == 0
    return ToolResult(id=call_id, ok=ok, output=output, error=None if ok else 'failed to merge PR')
