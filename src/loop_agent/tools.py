from __future__ import annotations

import os
import subprocess
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


def build_default_tools() -> dict[str, ToolFn]:
    return {
        'read_file': read_file_tool,
        'write_file': write_file_tool,
        'search': search_tool,
        'run_command': run_command_tool,
    }


def execute_tool_call(context: ToolContext, tool_call: ToolCall, tools: dict[str, ToolFn]) -> ToolResult:
    tool = tools.get(tool_call.name)
    if tool is None:
        return ToolResult(id=tool_call.id, ok=False, output='', error=f'unknown tool: {tool_call.name}')
    args = dict(tool_call.arguments)
    args.setdefault('id', tool_call.id)
    return tool(context, args)

