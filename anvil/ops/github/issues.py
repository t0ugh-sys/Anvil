from __future__ import annotations

import json
from typing import Dict, List

from ...agent.protocol import ToolResult
from ...tools import ToolContext
from ._shared import GhOptions, _merge_output, _require_gh_available, _resolve_repo_arg, _run_gh

__all__ = [
    'gh_issue_list_tool',
    'gh_issue_create_tool',
    'gh_issue_close_tool',
]


def gh_issue_list_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    call_id = str(args.get('id', 'gh_issue_list'))
    opts = GhOptions(cwd=str(context.workspace_root))

    repo_arg = str(args.get('repo', '')).strip()
    repo, err = _resolve_repo_arg(context, repo_arg)
    if err:
        return ToolResult(id=call_id, ok=False, output='', error=err)

    state = str(args.get('state', 'open')).strip().lower()
    limit = int(str(args.get('limit', '20')))
    if state not in {'open', 'closed', 'all'}:
        state = 'open'
    if limit <= 0:
        limit = 20

    missing = _require_gh_available(opts)
    if missing:
        return ToolResult(id=call_id, ok=False, output='', error=missing)

    proc = _run_gh(
        [
            'gh',
            'issue',
            'list',
            '--repo',
            str(repo),
            '--state',
            state,
            '--limit',
            str(limit),
            '--json',
            'number,title,url,state,author',
        ],
        opts=opts,
    )
    output = _merge_output(proc)
    if proc.returncode != 0:
        return ToolResult(id=call_id, ok=False, output=output, error='failed to list issues')

    try:
        data = json.loads(proc.stdout or '[]')
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output=proc.stdout or '', error=f'JSON parse error: {exc}')

    lines: List[str] = []
    for item in data:
        number = item.get('number')
        title = str(item.get('title') or '')
        url = str(item.get('url') or '')
        author = ''
        try:
            author = str((item.get('author') or {}).get('login') or '')
        except Exception:
            author = ''
        lines.append(f'#{number} {title} ({author}) {url}'.strip())

    return ToolResult(id=call_id, ok=True, output='\n'.join(lines), error=None)


def gh_issue_create_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    call_id = str(args.get('id', 'gh_issue_create'))
    opts = GhOptions(cwd=str(context.workspace_root))

    repo_arg = str(args.get('repo', '')).strip()
    repo, err = _resolve_repo_arg(context, repo_arg)
    if err:
        return ToolResult(id=call_id, ok=False, output='', error=err)

    title = str(args.get('title', '')).strip()
    body = str(args.get('body', '')).strip()

    if not title:
        return ToolResult(id=call_id, ok=False, output='', error='title is required')

    missing = _require_gh_available(opts)
    if missing:
        return ToolResult(id=call_id, ok=False, output='', error=missing)

    cmd: List[str] = ['gh', 'issue', 'create', '--repo', str(repo), '--title', title]
    if body:
        cmd.extend(['--body', body])

    proc = _run_gh(cmd, opts=opts)
    output = _merge_output(proc)
    ok = proc.returncode == 0
    return ToolResult(id=call_id, ok=ok, output=output, error=None if ok else 'failed to create issue')


def gh_issue_close_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    call_id = str(args.get('id', 'gh_issue_close'))
    opts = GhOptions(cwd=str(context.workspace_root))

    repo_arg = str(args.get('repo', '')).strip()
    repo, err = _resolve_repo_arg(context, repo_arg)
    if err:
        return ToolResult(id=call_id, ok=False, output='', error=err)

    number = str(args.get('number', '')).strip()
    confirm = bool(args.get('confirm', False))

    if not number:
        return ToolResult(id=call_id, ok=False, output='', error='number is required')
    if not confirm:
        return ToolResult(
            id=call_id,
            ok=False,
            output='',
            error='refusing to close issue without confirm=true',
        )

    missing = _require_gh_available(opts)
    if missing:
        return ToolResult(id=call_id, ok=False, output='', error=missing)

    proc = _run_gh(['gh', 'issue', 'close', number, '--repo', str(repo)], opts=opts)
    output = _merge_output(proc)
    ok = proc.returncode == 0
    return ToolResult(id=call_id, ok=ok, output=output, error=None if ok else 'failed to close issue')
