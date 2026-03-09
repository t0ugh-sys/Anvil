from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .agent_protocol import ToolResult
from .tools import ToolContext


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
    except Exception:
        return ToolResult(id=call_id, ok=True, output=output, error=None)

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
    except Exception:
        return ToolResult(id=call_id, ok=True, output=output, error=None)

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
    except Exception:
        return ToolResult(id=call_id, ok=True, output=output, error=None)

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
