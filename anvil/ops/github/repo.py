from __future__ import annotations

import json
from typing import Dict, List

from ...agent.protocol import ToolResult
from ...tools import ToolContext
from ._shared import GhOptions, _merge_output, _require_gh_available, _run_gh

__all__ = [
    'gh_auth_status_tool',
    'gh_repo_list_tool',
    'gh_repo_create_tool',
    'gh_repo_clone_tool',
]


def gh_auth_status_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    call_id = str(args.get('id', 'gh_auth_status'))
    opts = GhOptions(cwd=str(context.workspace_root))

    missing = _require_gh_available(opts)
    if missing:
        return ToolResult(id=call_id, ok=False, output='', error=missing)

    proc = _run_gh(['gh', 'auth', 'status'], opts=opts)
    output = _merge_output(proc)
    ok = proc.returncode == 0
    if ok:
        return ToolResult(id=call_id, ok=True, output=output, error=None)

    hint = 'Not authenticated. Run: gh auth login'
    return ToolResult(id=call_id, ok=False, output=output, error=hint)


def gh_repo_list_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    call_id = str(args.get('id', 'gh_repo_list'))
    opts = GhOptions(cwd=str(context.workspace_root))

    owner = str(args.get('owner', '')).strip()
    limit = int(str(args.get('limit', '30')))
    if limit <= 0:
        limit = 30

    missing = _require_gh_available(opts)
    if missing:
        return ToolResult(id=call_id, ok=False, output='', error=missing)

    cmd = ['gh', 'repo', 'list', '--limit', str(limit), '--json', 'name,owner,url,visibility,isPrivate']
    if owner:
        cmd.insert(3, owner)

    proc = _run_gh(cmd, opts=opts)
    output = _merge_output(proc)
    if proc.returncode != 0:
        return ToolResult(id=call_id, ok=False, output=output, error='failed to list repos')

    try:
        data = json.loads(proc.stdout or '[]')
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output=proc.stdout or '', error=f'JSON parse error: {exc}')

    lines: List[str] = []
    for item in data:
        owner_login = ''
        try:
            owner_login = str((item.get('owner') or {}).get('login') or '')
        except Exception:
            owner_login = ''
        name = str(item.get('name') or '')
        url = str(item.get('url') or '')
        visibility = str(item.get('visibility') or '')
        private = bool(item.get('isPrivate') or False)
        tag = visibility or ('private' if private else 'public')
        full = f'{owner_login}/{name}' if owner_login else name
        lines.append(f'{full} [{tag}] {url}'.strip())

    return ToolResult(id=call_id, ok=True, output='\n'.join(lines), error=None)


def gh_repo_create_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    call_id = str(args.get('id', 'gh_repo_create'))
    opts = GhOptions(cwd=str(context.workspace_root))

    name = str(args.get('name', '')).strip()
    visibility = str(args.get('visibility', 'private')).strip().lower()
    description = str(args.get('description', '')).strip()
    add_readme = bool(args.get('add_readme', True))

    if not name:
        return ToolResult(id=call_id, ok=False, output='', error='name is required')
    if visibility not in {'private', 'public', 'internal'}:
        visibility = 'private'

    missing = _require_gh_available(opts)
    if missing:
        return ToolResult(id=call_id, ok=False, output='', error=missing)

    cmd: List[str] = ['gh', 'repo', 'create', name]
    cmd.append(f'--{visibility}')
    cmd.append('--confirm')
    if description:
        cmd.extend(['--description', description])
    if add_readme:
        cmd.append('--add-readme')

    proc = _run_gh(cmd, opts=opts)
    output = _merge_output(proc)
    ok = proc.returncode == 0
    return ToolResult(id=call_id, ok=ok, output=output, error=None if ok else 'failed to create repo')


def gh_repo_clone_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    call_id = str(args.get('id', 'gh_repo_clone'))
    opts = GhOptions(cwd=str(context.workspace_root))

    repo = str(args.get('repo', '')).strip()
    dest = str(args.get('dest', '')).strip()

    if not repo:
        return ToolResult(id=call_id, ok=False, output='', error='repo is required (owner/name)')

    missing = _require_gh_available(opts)
    if missing:
        return ToolResult(id=call_id, ok=False, output='', error=missing)

    cmd: List[str] = ['gh', 'repo', 'clone', repo]
    if dest:
        cmd.append(dest)

    proc = _run_gh(cmd, opts=opts)
    output = _merge_output(proc)
    ok = proc.returncode == 0
    return ToolResult(id=call_id, ok=ok, output=output, error=None if ok else 'failed to clone repo')
