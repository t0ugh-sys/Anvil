from __future__ import annotations

import sys
from typing import Any, Dict

from .chrome import (
    DIM, GREEN, RED,
    CHECK_MARK, CROSS_MARK,
    colorize, truncate,
)

__all__ = ['render_tool_call', 'print_tool_call']

# ⎿  U+23BF — matches Claude Code's tool-call marker
_RESULT_MARKER = '⎿'

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
    label = colorize(_TOOL_LABELS.get(tool_name, '[?]'), DIM, enabled=color)
    summary = _summarize_args(tool_name, args)

    # Line 1 — "  ⎿ [r] read_file  README.md"
    header = f'  {_RESULT_MARKER} {label} {tool_name}'
    if summary:
        header += f'  {summary}'

    # Line 2 — "    ✓ 0.00s  4KB"
    if result.ok:
        size = len(result.output or '')
        size_str = f'{size}B' if size < 1024 else f'{size // 1024}KB'
        status = colorize(f'{CHECK_MARK} {elapsed_s:.2f}s  {size_str}', GREEN, enabled=color)
    else:
        err = truncate(result.error or 'failed', 48)
        status = colorize(f'{CROSS_MARK} {elapsed_s:.2f}s  {err}', RED, enabled=color)

    return [header, f'    {status}']


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
    for i, line in enumerate(rendered):
        # Clear the spinner line before the first write so the box doesn't
        # appear on the same line as "Working..."
        prefix = '\r\033[2K' if i == 0 else ''
        try:
            out.write(prefix + line + '\n')
        except UnicodeEncodeError:
            out.write((prefix + line).encode(enc, errors='replace').decode(enc) + '\n')
    out.flush()
