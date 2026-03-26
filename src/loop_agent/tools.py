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
from typing import Callable, Dict, Iterable, List, Tuple

from .agent_protocol import ToolCall, ToolResult
from .policies import ToolPolicy


@dataclass(frozen=True)
class ToolContext:
    workspace_root: Path
    policy: ToolPolicy = ToolPolicy.allow_all()


ToolFn = Callable[[ToolContext, Dict[str, object]], ToolResult]
ToolDispatchMap = Dict[str, ToolFn]
ToolRegistration = Tuple[str, ToolFn]


_SEARCH_SKIP_DIRS = {
    '.git',
    '.loopagent',
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
    root = workspace_root.resolve()
    target = (workspace_root / relative_path).resolve()
    if os.path.commonpath([str(root), str(target)]) != str(root):
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
    """Run a command in the workspace. 
    
    For security, prefer using 'cmd' (list) over 'command' (string with shell=True).
    If using 'command' string, ensure input is validated to prevent injection.
    """
    command = str(args.get('command', '')).strip()
    cmd_list = args.get('cmd')  # List of args for shell=False mode
    call_id = str(args.get('id', 'run_command'))
    
    # Determine if we're using safe mode (list of args) or legacy mode (string with shell)
    if cmd_list is not None and isinstance(cmd_list, list):
        # Safe mode: use list of arguments, shell=False
        try:
            proc = subprocess.run(
                cmd_list,
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
    
    # Legacy mode: string command with shell=True (DEPRECATED - security risk)
    if not command:
        return ToolResult(id=call_id, ok=False, output='', error='command or cmd is required')

    try:
        proc = subprocess.run(
            command,
            cwd=str(context.workspace_root),
            shell=True,
            check=False,
            text=True,
            capture_output=True,
            encoding='utf-8',
            errors='replace',
        )
        merged = (proc.stdout or '') + (proc.stderr or '')
        ok = proc.returncode == 0
        return ToolResult(id=call_id, ok=ok, output=merged.strip(), error=None if ok else f'exit={proc.returncode}')
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
            results.append(f'{i}. {title_clean}\n   URL: {link}\n   {snippet_clean[:200]}')
        
        if not results:
            return ToolResult(id=call_id, ok=True, output='No results found', error=None)
        
        output = f'Found {len(results)} results:\n\n' + '\n\n'.join(results)
        return ToolResult(id=call_id, ok=True, output=output, error=None)
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def fetch_url_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    """Fetch content from a specific URL."""
    url = str(args.get('url', '')).strip()
    call_id = str(args.get('id', 'fetch_url'))
    if not url:
        return ToolResult(id=call_id, ok=False, output='', error='url is required')

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
        max_chars = 5000
        if len(text) > max_chars:
            text = text[:max_chars] + f'\n\n... (truncated, total {len(text)} chars)'
        
        return ToolResult(id=call_id, ok=True, output=text.strip(), error=None)
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def analyze_memory_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    """Analyze past runs from memory store to learn patterns and insights.
    
    Args:
        memory_dir: Path to the memory store directory (default: .loopagent/runs)
        goal_filter: Optional goal to filter runs by
        limit: Number of recent runs to analyze (default: 5)
    """
    call_id = str(args.get('id', 'analyze_memory'))
    memory_dir = str(args.get('memory_dir', '.loopagent/runs'))
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


def register_tool_handler(dispatch_map: ToolDispatchMap, name: str, handler: ToolFn) -> ToolDispatchMap:
    dispatch_map[name] = handler
    return dispatch_map


def _build_tool_dispatch_map(registrations: Iterable[ToolRegistration]) -> ToolDispatchMap:
    dispatch_map: ToolDispatchMap = {}
    for name, handler in registrations:
        register_tool_handler(dispatch_map, name, handler)
    return dispatch_map


def _builtin_core_tool_registrations() -> List[ToolRegistration]:
    return [
        ('read_file', read_file_tool),
        ('write_file', write_file_tool),
        ('apply_patch', apply_patch_tool),
        ('search', search_tool),
        ('run_command', run_command_tool),
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
    return _build_tool_dispatch_map(registrations)


def execute_tool_call(context: ToolContext, tool_call: ToolCall, tools: ToolDispatchMap) -> ToolResult:
    tool = tools.get(tool_call.name)
    if tool is None:
        return ToolResult(id=tool_call.id, ok=False, output='', error=f'unknown tool: {tool_call.name}')
    if not context.policy.allows_tool(tool_call.name):
        denied = ', '.join(item.value for item in context.policy.denied_capabilities_for_tool(tool_call.name))
        return ToolResult(id=tool_call.id, ok=False, output='', error=f'tool blocked by policy: {tool_call.name} ({denied})')
    args = dict(tool_call.arguments)
    args.setdefault('id', tool_call.id)
    return tool(context, args)
