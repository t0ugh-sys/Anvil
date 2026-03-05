from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .agent_protocol import ToolCall, ToolResult


@dataclass(frozen=True)
class ToolContext:
    workspace_root: Path


ToolFn = Callable[[ToolContext, dict[str, object]], ToolResult]


def _resolve_inside_workspace(workspace_root: Path, relative_path: str) -> Path:
    root = workspace_root.resolve()
    target = (workspace_root / relative_path).resolve()
    if os.path.commonpath([str(root), str(target)]) != str(root):
        raise ValueError('path escapes workspace root')
    return target


def read_file_tool(context: ToolContext, args: dict[str, object]) -> ToolResult:
    path = str(args.get('path', ''))
    call_id = str(args.get('id', 'read_file'))
    try:
        target = _resolve_inside_workspace(context.workspace_root, path)
        content = target.read_text(encoding='utf-8')
        return ToolResult(id=call_id, ok=True, output=content)
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def write_file_tool(context: ToolContext, args: dict[str, object]) -> ToolResult:
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


def _split_patch_sections(patch_text: str) -> list[list[str]]:
    lines = patch_text.replace('\r\n', '\n').split('\n')
    sections: list[list[str]] = []
    current: list[str] = []
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


def _apply_update_hunks(origin: str, body_lines: list[str]) -> str:
    source_lines = origin.split('\n')
    cursor = 0
    index = 0

    while index < len(body_lines):
        line = body_lines[index]
        if not line.startswith('@@'):
            index += 1
            continue
        index += 1
        old_chunk: list[str] = []
        new_chunk: list[str] = []
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


def apply_patch_tool(context: ToolContext, args: dict[str, object]) -> ToolResult:
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
        changed: list[str] = []
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


def search_tool(context: ToolContext, args: dict[str, object]) -> ToolResult:
    pattern = str(args.get('pattern', '')).strip()
    call_id = str(args.get('id', 'search'))
    if not pattern:
        return ToolResult(id=call_id, ok=False, output='', error='pattern is required')

    try:
        results: list[str] = []
        for path in context.workspace_root.rglob('*'):
            if not path.is_file():
                continue
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


def run_command_tool(context: ToolContext, args: dict[str, object]) -> ToolResult:
    command = str(args.get('command', '')).strip()
    call_id = str(args.get('id', 'run_command'))
    if not command:
        return ToolResult(id=call_id, ok=False, output='', error='command is required')

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


def web_search_tool(context: ToolContext, args: dict[str, object]) -> ToolResult:
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
        results: list[str] = []
        import re
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


def fetch_url_tool(context: ToolContext, args: dict[str, object]) -> ToolResult:
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
        import re
        # Remove script and style elements
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Remove all HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Clean up whitespace
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Limit output size
        max_chars = int(args.get('max_chars', 5000))
        if len(text) > max_chars:
            text = text[:max_chars] + f'\n\n... (truncated, total {len(text)} chars)'
        
        return ToolResult(id=call_id, ok=True, output=text.strip(), error=None)
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def build_default_tools() -> dict[str, ToolFn]:
    return {
        'read_file': read_file_tool,
        'write_file': write_file_tool,
        'apply_patch': apply_patch_tool,
        'search': search_tool,
        'run_command': run_command_tool,
        'web_search': web_search_tool,
        'fetch_url': fetch_url_tool,
    }


def execute_tool_call(context: ToolContext, tool_call: ToolCall, tools: dict[str, ToolFn]) -> ToolResult:
    tool = tools.get(tool_call.name)
    if tool is None:
        return ToolResult(id=tool_call.id, ok=False, output='', error=f'unknown tool: {tool_call.name}')
    args = dict(tool_call.arguments)
    args.setdefault('id', tool_call.id)
    return tool(context, args)
