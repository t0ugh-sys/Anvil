from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Tuple

from .agent_protocol import ToolCall, ToolResult
from .background import BackgroundCommandRunner
from .compression import CompactManager
from .permissions import PermissionMode
from .policies import ToolPolicy
from .policies import TOOL_CAPABILITIES
from .tool_spec import ToolDef, ToolRisk, ToolSpec, ValidationResult, build_tool
from .todo import TodoManager, render_todo_lines

if TYPE_CHECKING:
    from .skills import SkillLoader

# Constants
MAX_FETCH_URL_CHARS = 5000
MAX_SEARCH_SNIPPET_CHARS = 200
MAX_WEB_RESULT_TITLE_CHARS = 120


def require_params(*names: str) -> Callable[[Dict[str, object]], ValidationResult]:
    """Create a validator that checks for required parameters."""
    def validate(args: Dict[str, object]) -> ValidationResult:
        missing = [n for n in names if not args.get(n)]
        if missing:
            return ValidationResult.failure(
                *(f'required parameter `{n}` is missing' for n in missing)
            )
        return ValidationResult.success()
    return validate


@dataclass(frozen=True)
class ToolContext:
    workspace_root: Path
    policy: ToolPolicy = ToolPolicy.allow_all()
    todo_manager: TodoManager | None = None
    skill_loader: SkillLoader | None = None
    compact_manager: CompactManager | None = None
    background_runner: BackgroundCommandRunner | None = None


ToolFn = Callable[[ToolContext, Dict[str, object]], ToolResult]
ToolDispatchMap = Dict[str, ToolFn]
ToolRegistration = Tuple[str, ToolFn]

# Registry of full tool definitions (for validation, metadata)
_tool_defs: Dict[str, ToolDef] = {}


_SEARCH_SKIP_DIRS = {
    '.git',
    '.anvil',
    '.mypy_cache',
    '.pytest_cache',
    '.ruff_cache',
    '.venv',
    '__pycache__',
    'build',
    'dist',
    'node_modules',
}


def _iter_searchable_files(workspace_root: Path) -> Iterable[Path]:
    for current_root, dir_names, file_names in os.walk(workspace_root):
        dir_names[:] = [name for name in dir_names if name not in _SEARCH_SKIP_DIRS]
        root_path = Path(current_root)
        for file_name in file_names:
            yield root_path / file_name


def _resolve_inside_workspace(workspace_root: Path, relative_path: str) -> Path:
    if '\0' in relative_path:
        raise ValueError('path contains null bytes')
    if any(part == '..' for part in relative_path.replace('\\', '/').split('/')):
        raise ValueError('path escapes workspace root')
    root = workspace_root.resolve()
    target = (workspace_root / relative_path).resolve()
    if not target.is_relative_to(root):
        raise ValueError('path escapes workspace root')
    return target


def read_file_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    path = str(args.get('path', ''))
    call_id = str(args.get('id', 'read_file'))
    try:
        target = _resolve_inside_workspace(context.workspace_root, path)
        content = target.read_text(encoding='utf-8')
        return ToolResult(id=call_id, ok=True, output=content)
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def write_file_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    path = str(args.get('path', ''))
    content = str(args.get('content', ''))
    call_id = str(args.get('id', 'write_file'))
    try:
        target = _resolve_inside_workspace(context.workspace_root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
        return ToolResult(id=call_id, ok=True, output='ok')
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def _split_patch_sections(patch_text: str) -> List[List[str]]:
    lines = patch_text.replace('\r\n', '\n').split('\n')
    sections: List[List[str]] = []
    current: List[str] = []
    for line in lines:
        if line.startswith('*** ') and current:
            sections.append(current)
            current = [line]
            continue
        current.append(line)
    if current:
        sections.append(current)
    return sections


def _resolve_patch_target(context: ToolContext, header: str) -> Path:
    raw = header.split(':', 1)[1].strip()
    if not raw:
        raise ValueError('patch target path is empty')
    return _resolve_inside_workspace(context.workspace_root, raw)


def _apply_update_hunks(origin: str, body_lines: List[str]) -> str:
    source_lines = origin.split('\n')
    cursor = 0
    index = 0

    while index < len(body_lines):
        line = body_lines[index]
        if not line.startswith('@@'):
            index += 1
            continue
        index += 1
        old_chunk: List[str] = []
        new_chunk: List[str] = []
        while index < len(body_lines):
            current = body_lines[index]
            if current.startswith('@@'):
                break
            if not current:
                old_chunk.append('')
                new_chunk.append('')
                index += 1
                continue
            marker = current[:1]
            value = current[1:]
            if marker == ' ':
                old_chunk.append(value)
                new_chunk.append(value)
            elif marker == '-':
                old_chunk.append(value)
            elif marker == '+':
                new_chunk.append(value)
            else:
                raise ValueError(f'unsupported patch marker: {marker}')
            index += 1

        if old_chunk:
            found = -1
            max_start = len(source_lines) - len(old_chunk)
            for start in range(cursor, max_start + 1):
                if source_lines[start : start + len(old_chunk)] == old_chunk:
                    found = start
                    break
            if found < 0:
                raise ValueError('patch hunk does not match file content')
            source_lines = source_lines[:found] + new_chunk + source_lines[found + len(old_chunk) :]
            cursor = found + len(new_chunk)
        else:
            source_lines = source_lines[:cursor] + new_chunk + source_lines[cursor:]
            cursor = cursor + len(new_chunk)

    return '\n'.join(source_lines)


def apply_patch_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    patch_text = str(args.get('patch', ''))
    call_id = str(args.get('id', 'apply_patch'))
    if not patch_text.strip():
        return ToolResult(id=call_id, ok=False, output='', error='patch is required')

    try:
        root = context.workspace_root.resolve()
        normalized = patch_text.replace('\r\n', '\n').strip('\n')
        if not normalized.startswith('*** Begin Patch') or not normalized.endswith('*** End Patch'):
            raise ValueError('patch must start with "*** Begin Patch" and end with "*** End Patch"')
        content = normalized[len('*** Begin Patch') : -len('*** End Patch')].strip('\n')
        if not content:
            raise ValueError('patch body is empty')

        sections = _split_patch_sections(content)
        changed: List[str] = []
        for section in sections:
            header = section[0]
            body = section[1:]
            if header.startswith('*** Add File:'):
                target = _resolve_patch_target(context, header)
                if target.exists():
                    raise ValueError(f'file already exists: {target}')
                add_lines = [line[1:] for line in body if line.startswith('+')]
                if len(add_lines) != len(body):
                    raise ValueError('add file section only supports + lines')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('\n'.join(add_lines), encoding='utf-8')
                changed.append(target.relative_to(root).as_posix())
                continue

            if header.startswith('*** Delete File:'):
                target = _resolve_patch_target(context, header)
                if target.exists():
                    target.unlink()
                changed.append(target.relative_to(root).as_posix())
                continue

            if header.startswith('*** Update File:'):
                target = _resolve_patch_target(context, header)
                if not target.exists():
                    raise ValueError(f'file not found: {target}')
                original = target.read_text(encoding='utf-8')
                updated = _apply_update_hunks(original, body)
                target.write_text(updated, encoding='utf-8')
                changed.append(target.relative_to(root).as_posix())
                continue

            raise ValueError(f'unsupported patch header: {header}')

        return ToolResult(id=call_id, ok=True, output='\n'.join(changed))
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def search_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    pattern = str(args.get('pattern', '')).strip()
    call_id = str(args.get('id', 'search'))
    if not pattern:
        return ToolResult(id=call_id, ok=False, output='', error='pattern is required')

    try:
        # Try grep first (fast), fallback to Python search
        try:
            result = subprocess.run(
                ['grep', '-rl', '--include=*.py', '--include=*.md', '--include=*.txt',
                 '--include=*.json', '--include=*.yaml', '--include=*.yml',
                 '--include=*.toml', '--include=*.js', '--include=*.ts',
                 pattern, '.'],
                cwd=str(context.workspace_root),
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                files = sorted(result.stdout.strip().split('\n'))
                # Strip ./ prefix from grep output
                files = [f[2:] if f.startswith('./') else f for f in files]
                return ToolResult(id=call_id, ok=True, output='\n'.join(files))
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Fallback: Python-based search
        results: List[str] = []
        for path in _iter_searchable_files(context.workspace_root):
            try:
                text = path.read_text(encoding='utf-8')
            except Exception:
                continue
            if pattern in text:
                relative = path.relative_to(context.workspace_root).as_posix()
                results.append(relative)
        return ToolResult(id=call_id, ok=True, output='\n'.join(results))
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def run_command_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    """Run a command in the workspace using shell=False mode for security.

    Args:
        cmd: List of command arguments (e.g., ['ls', '-la'])
        id: Optional tool call ID

    Note: Shell features like pipes and wildcards are not supported.
          Use 'bash -c "cmd1 | cmd2"' if shell features are needed.
    """
    cmd_list = args.get('cmd')
    call_id = str(args.get('id', 'run_command'))

    if not isinstance(cmd_list, list):
        return ToolResult(id=call_id, ok=False, output='', error='cmd list is required')

    try:
        proc = subprocess.run(
            [str(item) for item in cmd_list],
            cwd=str(context.workspace_root),
            shell=False,
            check=False,
            text=True,
            capture_output=True,
            encoding='utf-8',
            errors='replace',
        )
        merged = (proc.stdout or '') + (proc.stderr or '')
        ok = proc.returncode == 0
        return ToolResult(id=call_id, ok=ok, output=merged.strip(), error=None if ok else f'exit={proc.returncode}')
    except FileNotFoundError as exc:
        return ToolResult(id=call_id, ok=False, output='', error=f'command not found: {exc.filename}')
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def web_search_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    """Search the web using DuckDuckGo HTML search (no API key required)."""
    query = str(args.get('query', '')).strip()
    call_id = str(args.get('id', 'web_search'))
    if not query:
        return ToolResult(id=call_id, ok=False, output='', error='query is required')

    try:
        # Use DuckDuckGo HTML search
        encoded_query = urllib.parse.quote_plus(query)
        url = f'https://html.duckduckgo.com/html/?q={encoded_query}'
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        request = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(request, timeout=30) as response:
            html = response.read().decode('utf-8', errors='replace')
        
        # Parse results from HTML
        results: List[str] = []
        # Match result blocks
        pattern = r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.+?)</a>.*?<a class="result__snippet"[^>]*>(.+?)</a>'
        matches = re.findall(pattern, html, re.DOTALL)
        
        for i, (link, title, snippet) in enumerate(matches[:10], 1):
            # Clean up HTML tags from title and snippet
            title_clean = re.sub(r'<[^>]+>', '', title).strip()
            snippet_clean = re.sub(r'<[^>]+>', '', snippet).strip()
            results.append(f'{i}. {title_clean}\n   URL: {link}\n   {snippet_clean[:MAX_SEARCH_SNIPPET_CHARS]}')
        
        if not results:
            return ToolResult(id=call_id, ok=True, output='No results found', error=None)
        
        output = f'Found {len(results)} results:\n\n' + '\n\n'.join(results)
        return ToolResult(id=call_id, ok=True, output=output, error=None)
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


_ALLOWED_URL_SCHEMES = frozenset({'http', 'https'})
_BLOCKED_URL_HOSTS = frozenset({
    '169.254.169.254',  # AWS/GCP/Azure metadata
    'metadata.google.internal',
    'localhost',
    '127.0.0.1',
    '::1',
})


def fetch_url_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    """Fetch content from a specific URL."""
    url = str(args.get('url', '')).strip()
    call_id = str(args.get('id', 'fetch_url'))
    if not url:
        return ToolResult(id=call_id, ok=False, output='', error='url is required')

    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme not in _ALLOWED_URL_SCHEMES:
            return ToolResult(id=call_id, ok=False, output='', error=f'blocked URL scheme: {parsed.scheme}')
        if parsed.hostname in _BLOCKED_URL_HOSTS:
            return ToolResult(id=call_id, ok=False, output='', error='blocked URL host')
    except Exception:
        return ToolResult(id=call_id, ok=False, output='', error='invalid URL')

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        request = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(request, timeout=30) as response:
            html = response.read().decode('utf-8', errors='replace')
        
        # Simple HTML to text conversion - remove scripts, styles, and tags
        # Remove script and style elements
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Remove all HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Clean up whitespace
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Limit output size
        if len(text) > MAX_FETCH_URL_CHARS:
            text = text[:MAX_FETCH_URL_CHARS] + f'\n\n... (truncated, total {len(text)} chars)'
        
        return ToolResult(id=call_id, ok=True, output=text.strip(), error=None)
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def analyze_memory_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    """Analyze past runs from memory store to learn patterns and insights.
    
    Args:
        memory_dir: Path to the memory store directory (default: .anvil/runs)
        goal_filter: Optional goal to filter runs by
        limit: Number of recent runs to analyze (default: 5)
    """
    call_id = str(args.get('id', 'analyze_memory'))
    memory_dir = str(args.get('memory_dir', '.anvil/runs'))
    goal_filter = str(args.get('goal_filter', '')).strip()
    limit = int(str(args.get('limit', '5')))
    
    try:
        memory_path = Path(memory_dir)
        if not memory_path.exists():
            return ToolResult(id=call_id, ok=False, output='', error=f'memory directory not found: {memory_dir}')
        
        # Find all run directories
        run_dirs = sorted([d for d in memory_path.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
        run_dirs = run_dirs[:limit]
        
        if not run_dirs:
            return ToolResult(id=call_id, ok=True, output='No past runs found in memory', error=None)
        
        analysis: List[str] = []
        total_runs = 0
        completed_runs = 0
        failed_runs = 0
        
        for run_dir in run_dirs:
            summary_file = run_dir / 'summary.json'
            if not summary_file.exists():
                continue
            
            try:
                with summary_file.open(encoding='utf-8') as f:
                    summary = json.load(f)
                
                goal = summary.get('goal', '')
                if goal_filter and goal_filter.lower() not in goal.lower():
                    continue
                
                total_runs += 1
                done = summary.get('done', False)
                stop_reason = summary.get('stop_reason', 'unknown')
                steps = summary.get('steps', 0)
                
                if done:
                    completed_runs += 1
                else:
                    failed_runs += 1
                
                analysis.append(f'Run: {run_dir.name}')
                analysis.append(f'  Goal: {goal[:80]}...' if len(goal) > 80 else f'  Goal: {goal}')
                analysis.append(f'  Result: {"✓ Completed" if done else "✗ Failed"} (stop: {stop_reason})')
                analysis.append(f'  Steps: {steps}')
                analysis.append('')
            except Exception:
                continue
        
        # Build summary
        summary_text = [
            f'=== Memory Analysis (Last {limit} runs) ===',
            f'Total runs analyzed: {total_runs}',
            f'Completed: {completed_runs}',
            f'Failed: {failed_runs}',
            f'Success rate: {completed_runs/total_runs*100:.1f}%' if total_runs > 0 else 'N/A',
            '',
            '--- Recent Runs ---',
            ''
        ]
        summary_text.extend(analysis)
        
        return ToolResult(id=call_id, ok=True, output='\n'.join(summary_text), error=None)
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def run_command_async_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    call_id = str(args.get('id', 'run_command_async'))
    runner = context.background_runner
    if runner is None:
        return ToolResult(id=call_id, ok=False, output='', error='background runner is not configured')

    cmd_list = args.get('cmd')
    if not isinstance(cmd_list, list):
        return ToolResult(id=call_id, ok=False, output='', error='cmd list is required')
    return runner.spawn(command=[str(item) for item in cmd_list], call_id=call_id)


def todo_write_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    call_id = str(args.get('id', 'todo_write'))
    manager = context.todo_manager
    if manager is None:
        return ToolResult(id=call_id, ok=False, output='', error='todo manager is not configured')

    items = args.get('items')
    if not isinstance(items, list):
        return ToolResult(id=call_id, ok=False, output='', error='items list is required')

    try:
        updated_items = manager.write(items)
        lines = [
            'todo updated',
            *[f'- {line}' for line in render_todo_lines(updated_items)],
        ]
        return ToolResult(id=call_id, ok=True, output='\n'.join(lines), error=None)
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def load_skill_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    call_id = str(args.get('id', 'load_skill'))
    loader = context.skill_loader
    if loader is None:
        return ToolResult(id=call_id, ok=False, output='', error='skill loader is not configured')

    name = str(args.get('name', '')).strip()
    if not name:
        return ToolResult(id=call_id, ok=False, output='', error='skill name is required')

    body = loader.load_body(name)
    if body is None:
        return ToolResult(id=call_id, ok=False, output='', error=f'skill not loaded: {name}')
    return ToolResult(id=call_id, ok=True, output=f'<skill name=\"{name}\">\n{body}\n</skill>')


def compact_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    call_id = str(args.get('id', 'compact'))
    manager = context.compact_manager
    if manager is None:
        return ToolResult(id=call_id, ok=False, output='', error='compact manager is not configured')

    reason = str(args.get('reason', '')).strip()
    manager.request(reason)
    message = 'compaction requested'
    if reason:
        message += f': {reason}'
    return ToolResult(id=call_id, ok=True, output=message, error=None)


_BUILTIN_TOOL_DESCRIPTIONS: Dict[str, str] = {
    'load_skill': 'Load full skill instructions into the conversation on demand.',
    'compact': 'Request transcript compaction for long-running sessions.',
    'todo_write': 'Update the visible todo list stored in runtime state.',
    'read_file': 'Read one file from the workspace.',
    'write_file': 'Write one UTF-8 file inside the workspace.',
    'apply_patch': 'Apply a structured patch to workspace files.',
    'search': 'Search workspace files for a literal pattern.',
    'run_command': 'Run one command with shell disabled inside the workspace.',
    'run_command_async': 'Start one background command inside the workspace.',
    'web_search': 'Search the web through a simple HTTP search endpoint.',
    'fetch_url': 'Fetch and strip text content from one URL.',
    'analyze_memory': 'Analyze prior run summaries under the memory directory.',
    'git_status': 'Inspect git repository status.',
    'git_branch_list': 'List git branches.',
    'git_checkout': 'Switch git branches.',
    'git_pull': 'Pull from a remote branch.',
    'git_merge': 'Merge git branches.',
    'git_merge_and_push': 'Merge changes and push them upstream.',
    'git_push': 'Push local branch state upstream.',
    'gh_auth_status': 'Check GitHub CLI authentication state.',
    'gh_repo_list': 'List GitHub repositories.',
    'gh_repo_create': 'Create a GitHub repository.',
    'gh_repo_clone': 'Clone a GitHub repository into the workspace.',
    'gh_issue_list': 'List GitHub issues.',
    'gh_issue_create': 'Create a GitHub issue.',
    'gh_issue_close': 'Close a GitHub issue.',
    'gh_pr_list': 'List GitHub pull requests.',
    'gh_pr_create': 'Create a GitHub pull request.',
    'gh_pr_view': 'View a GitHub pull request.',
    'gh_pr_checks': 'Inspect GitHub pull request checks.',
    'gh_pr_comment': 'Comment on a GitHub pull request.',
    'gh_pr_merge': 'Merge a GitHub pull request.',
}

_BUILTIN_TOOL_INPUT_NOTES: Dict[str, str] = {
    'apply_patch': 'Provide a full structured patch with Begin/End Patch markers.',
    'run_command': 'Use cmd as a list of arguments; shell syntax is not supported.',
    'run_command_async': 'Use cmd as a list of arguments; command runs in background.',
    'web_search': 'Provide query text.',
    'fetch_url': 'Provide one URL string.',
    'todo_write': 'Provide items as a list of todo objects.',
}


def _risk_for_capabilities(capabilities: Tuple[Any, ...]) -> ToolRisk:
    if any(item.value in {'write', 'network'} for item in capabilities):
        return ToolRisk.high
    if any(item.value == 'execute' for item in capabilities):
        return ToolRisk.high
    if any(item.value == 'memory' for item in capabilities):
        return ToolRisk.medium
    return ToolRisk.low


def builtin_tool_specs() -> List[ToolSpec]:
    names = [name for name, _ in builtin_tool_registrations()]
    specs: List[ToolSpec] = []
    for name in names:
        capabilities = TOOL_CAPABILITIES.get(name, tuple())
        specs.append(
            ToolSpec(
                name=name,
                description=_BUILTIN_TOOL_DESCRIPTIONS.get(name, f'{name} tool'),
                capabilities=capabilities,
                risk_level=_risk_for_capabilities(capabilities),
                requires_workspace=True,
                input_notes=_BUILTIN_TOOL_INPUT_NOTES.get(name, ''),
            )
        )
    return specs


def builtin_tool_specs_map() -> Dict[str, ToolSpec]:
    return {item.name: item for item in builtin_tool_specs()}
def register_tool_handler(dispatch_map: ToolDispatchMap, name: str, handler: ToolFn) -> ToolDispatchMap:
    dispatch_map[name] = handler
    return dispatch_map


def _build_tool_dispatch_map(registrations: Iterable[ToolRegistration]) -> ToolDispatchMap:
    dispatch_map: ToolDispatchMap = {}
    for name, handler in registrations:
        register_tool_handler(dispatch_map, name, handler)
    return dispatch_map


def register_tool_def(tool_def: ToolDef) -> None:
    """Register a full tool definition for validation and metadata."""
    _tool_defs[tool_def.name] = tool_def


def get_tool_def(name: str) -> ToolDef | None:
    """Get the full tool definition for a tool by name."""
    return _tool_defs.get(name)


def _builtin_core_tool_registrations() -> List[ToolRegistration]:
    return [
        ('load_skill', load_skill_tool),
        ('compact', compact_tool),
        ('todo_write', todo_write_tool),
        ('read_file', read_file_tool),
        ('write_file', write_file_tool),
        ('apply_patch', apply_patch_tool),
        ('search', search_tool),
        ('run_command', run_command_tool),
        ('run_command_async', run_command_async_tool),
        ('web_search', web_search_tool),
        ('fetch_url', fetch_url_tool),
        ('analyze_memory', analyze_memory_tool),
    ]


def _builtin_git_tool_registrations() -> List[ToolRegistration]:
    from .ops.git_tools import (
        git_branch_list_tool,
        git_checkout_tool,
        git_merge_and_push_tool,
        git_merge_tool,
        git_pull_tool,
        git_push_tool,
        git_status_tool,
    )

    return [
        ('git_status', git_status_tool),
        ('git_branch_list', git_branch_list_tool),
        ('git_checkout', git_checkout_tool),
        ('git_pull', git_pull_tool),
        ('git_merge', git_merge_tool),
        ('git_merge_and_push', git_merge_and_push_tool),
        ('git_push', git_push_tool),
    ]


def _builtin_github_tool_registrations() -> List[ToolRegistration]:
    from .ops.github_tools import (
        gh_auth_status_tool,
        gh_issue_close_tool,
        gh_issue_create_tool,
        gh_issue_list_tool,
        gh_pr_checks_tool,
        gh_pr_comment_tool,
        gh_pr_create_tool,
        gh_pr_list_tool,
        gh_pr_merge_tool,
        gh_pr_view_tool,
        gh_repo_clone_tool,
        gh_repo_create_tool,
        gh_repo_list_tool,
    )

    return [
        ('gh_auth_status', gh_auth_status_tool),
        ('gh_repo_list', gh_repo_list_tool),
        ('gh_repo_create', gh_repo_create_tool),
        ('gh_repo_clone', gh_repo_clone_tool),
        ('gh_issue_list', gh_issue_list_tool),
        ('gh_issue_create', gh_issue_create_tool),
        ('gh_issue_close', gh_issue_close_tool),
        ('gh_pr_list', gh_pr_list_tool),
        ('gh_pr_create', gh_pr_create_tool),
        ('gh_pr_view', gh_pr_view_tool),
        ('gh_pr_checks', gh_pr_checks_tool),
        ('gh_pr_comment', gh_pr_comment_tool),
        ('gh_pr_merge', gh_pr_merge_tool),
    ]


def builtin_tool_registrations() -> List[ToolRegistration]:
    # Keep tool names stable; they are part of the model <-> harness contract.
    # To add a new tool, register one more handler here. The loop itself stays unchanged.
    registrations: List[ToolRegistration] = []
    registrations.extend(_builtin_core_tool_registrations())
    registrations.extend(_builtin_git_tool_registrations())
    registrations.extend(_builtin_github_tool_registrations())
    return registrations


def build_default_tools() -> ToolDispatchMap:
    return _build_tool_dispatch_map(builtin_tool_registrations())



def execute_tool_call(context: ToolContext, tool_call: ToolCall, tools: ToolDispatchMap) -> ToolResult:
    tool = tools.get(tool_call.name)
    if tool is None:
        return ToolResult(id=tool_call.id, ok=False, output='', error=f'unknown tool: {tool_call.name}')

    # Phase 1: Validate input (if tool def has validator)
    tool_def = _tool_defs.get(tool_call.name)
    if tool_def and tool_def.validate_input:
        validation = tool_def.validate_input(tool_call.arguments)
        if not validation.ok:
            return ToolResult(
                id=tool_call.id,
                ok=False,
                output='',
                error=validation.to_error_string(tool_call.name),
            )

    # Phase 2: Check permissions
    if not context.policy.allows_tool(tool_call.name):
        denied = ', '.join(item.value for item in context.policy.denied_capabilities_for_tool(tool_call.name))
        return ToolResult(
            id=tool_call.id,
            ok=False,
            output='',
            error=f'tool blocked by policy: {tool_call.name} ({denied})',
            metadata={'permission_decision': 'deny', 'permission_reason': f'policy blocked capabilities: {denied}'},
        )
    permission_manager = getattr(context.policy, 'permission_manager', None)
    if permission_manager is not None:
        capabilities = TOOL_CAPABILITIES.get(tool_call.name, tuple())
        request = permission_manager.build_request(
            tool_name=tool_call.name,
            arguments=dict(tool_call.arguments),
            workspace_root=context.workspace_root,
            capabilities=capabilities,
        )
        decision = permission_manager.decide(request)
        if decision.mode != PermissionMode.allow:
            permission_manager.record_decision(decision.cache_key, decision.mode)
            return ToolResult(
                id=tool_call.id,
                ok=False,
                output='',
                error=f'tool blocked by permission: {tool_call.name} ({decision.reason})',
                metadata={
                    'permission_decision': decision.mode,
                    'permission_reason': decision.reason,
                    'permission_cached': decision.cached,
                },
            )
    args = dict(tool_call.arguments)
    args.setdefault('id', tool_call.id)
    result = tool(context, args)
    if permission_manager is not None:
        request = permission_manager.build_request(
            tool_name=tool_call.name,
            arguments=args,
            workspace_root=context.workspace_root,
            capabilities=TOOL_CAPABILITIES.get(tool_call.name, tuple()),
        )
        permission_manager.record_decision(request.cache_key, PermissionMode.allow)
        result.metadata.setdefault('permission_decision', PermissionMode.allow)
        result.metadata.setdefault('permission_reason', f'{tool_call.name} allowed')
    return result
