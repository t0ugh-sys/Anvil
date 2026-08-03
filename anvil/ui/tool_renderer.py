from __future__ import annotations

import sys
from typing import Any, Dict

from .chrome import (
    DIM, GREEN, RED, RESET,
    CHECK_MARK, CROSS_MARK,
    bounded_width, box_lines, colorize, truncate,
)

__all__ = ['render_tool_call', 'print_tool_call']

_TOOL_LABELS: Dict[str, str] = {
    'read_file': '[r]',
    'write_file': '[w]',
    'apply_patch': '[p]',
    'run_command': '[>]',
    'search_files': '[s]',
    'list_dir': '[d]',
    'memory': '[m]',
    'web_search': '[*]',
    'github_cli': '[g]',
    'todo_write': '[t]',
    'todo_reminder': '[t]',
}

_PATH_TOOLS = frozenset({'read_file', 'write_file', 'apply_patch'})
_CMD_TOOLS = frozenset({'run_command'})
_SEARCH_TOOLS = frozenset({'search_files'})


def _summarize_args(tool_name: str, args: Dict[str, Any], *, max_len: int = 64) -> str:
    if tool_name in _PATH_TOOLS:
        path = args.get('path') or args.get('file_path', '')
        return truncate(str(path), max_len) if path else ''
    if tool_name in _CMD_TOOLS:
        return truncate(str(args.get('command', '')), max_len)
    if tool_name in _SEARCH_TOOLS:
        val = args.get('pattern') or args.get('query', '')
        return truncate(str(val), max_len) if val else ''
    if tool_name == 'list_dir':
        return truncate(str(args.get('path', '.')), max_len)
    for v in args.values():
        return truncate(str(v), max_len)
    return ''


def render_tool_call(
    tool_name: str,
    args: Dict[str, Any],
    result: Any,
    elapsed_s: float,
    *,
    width: int = 80,
    color: bool = True,
) -> list[str]:
    label = _TOOL_LABELS.get(tool_name, '[?]')
    title = f'{label} {tool_name}'
    summary = _summarize_args(tool_name, args)

    if result.ok:
        size = len(result.output or '')
        size_str = f'{size}B' if size < 1024 else f'{size // 1024}KB'
        status = f'{CHECK_MARK} {elapsed_s:.2f}s  {size_str}'
        status_colored = colorize(status, GREEN, enabled=color)
    else:
        err = truncate((result.error or 'failed'), 48)
        status = f'{CROSS_MARK} {elapsed_s:.2f}s  {err}'
        status_colored = colorize(status, RED, enabled=color)

    content_lines: list[str] = []
    if summary:
        content_lines.append(colorize(summary, DIM, enabled=color))
    content_lines.append(status_colored)

    return box_lines(content_lines, width=width, title=title)


def print_tool_call(
    tool_name: str,
    args: Dict[str, Any],
    result: Any,
    elapsed_s: float,
    *,
    width: int = 80,
    color: bool = True,
) -> None:
    rendered = render_tool_call(tool_name, args, result, elapsed_s, width=width, color=color)
    out = sys.stdout
    enc = getattr(out, 'encoding', None) or 'utf-8'
    for line in rendered:
        try:
            out.write(line + '\n')
        except UnicodeEncodeError:
            out.write(line.encode(enc, errors='replace').decode(enc) + '\n')
    out.flush()
