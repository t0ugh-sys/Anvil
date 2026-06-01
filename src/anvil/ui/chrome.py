from __future__ import annotations

import textwrap
from typing import Iterable

__all__ = [
    'MIN_UI_WIDTH', 'MAX_UI_WIDTH',
    'colorize', 'bounded_width', 'truncate', 'wrap_line',
    'top_border', 'bottom_border', 'separator_line',
    'box_lines', 'response_lines', 'status_bar',
]

MIN_UI_WIDTH = 48
MAX_UI_WIDTH = 100

# Box drawing
TOP_LEFT = '\u256d'
TOP_RIGHT = '\u256e'
BOTTOM_LEFT = '\u2570'
BOTTOM_RIGHT = '\u256f'
HORIZONTAL = '\u2500'
VERTICAL = '\u2502'
DOUBLE_HORIZONTAL = '\u2550'
DOUBLE_VERTICAL = '\u2551'

# Markers
RESPONSE_MARKER = '\u23bf'
WORKING_MARKER = '\u273b'
DOT_SEPARATOR = '\u00b7'
PROMPT_MARKER = '\u276f'
ARROW_RIGHT = '\u2192'
CHECK_MARK = '\u2713'
CROSS_MARK = '\u2717'
SPARKLE = '\u2728'
GEAR = '\u2699'
LIGHTNING = '\u26a1'
FOLDER = '\u2630'
BULLET = '\u2022'
DIAMOND = '\u25c6'
CIRCLE_FILLED = '\u25cf'
CIRCLE_EMPTY = '\u25cb'
BLOCK_LOWER = '\u2584'
LIGHT_SHADE = '\u2591'

# ANSI colors
RESET = '\033[0m'
BOLD = '\033[1m'
DIM = '\033[38;5;245m'
MUTED = '\033[38;5;244m'
BORDER = '\033[38;5;66m'
ACCENT = '\033[38;5;110m'
PROMPT = '\033[38;5;120m'
WORKING = '\033[38;5;116m'
ASSISTANT = '\033[38;5;252m'
RED = '\033[31m'
GREEN = '\033[38;5;82m'
YELLOW = '\033[38;5;220m'
CYAN = '\033[38;5;87m'
SUCCESS = '\033[38;5;114m'
WARNING = '\033[38;5;214m'
ERROR = '\033[38;5;203m'
INFO = '\033[38;5;75m'


def colorize(value: str, style: str, *, enabled: bool) -> str:
    if not enabled:
        return value
    return f'{style}{value}{RESET}'


def bounded_width(columns: int) -> int:
    return max(MIN_UI_WIDTH, min(MAX_UI_WIDTH, max(1, columns - 2)))


def truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[: width - 3] + '\u2026'


def wrap_line(value: str, width: int) -> list[str]:
    if not value:
        return ['']
    return textwrap.wrap(
        value,
        width=max(1, width),
        break_long_words=True,
        break_on_hyphens=False,
        replace_whitespace=False,
    ) or ['']


def _repeat(char: str, count: int) -> str:
    return char * max(0, count)


def top_border(width: int, *, title: str = '') -> str:
    inner_width = max(2, width - 2)
    if not title:
        return TOP_LEFT + _repeat(DOUBLE_HORIZONTAL, inner_width) + TOP_RIGHT
    label = truncate(f' {SPARKLE} {title} ', inner_width)
    left_pad = 2
    right_count = max(0, inner_width - left_pad - len(label))
    return TOP_LEFT + _repeat(DOUBLE_HORIZONTAL, left_pad) + label + _repeat(DOUBLE_HORIZONTAL, right_count) + TOP_RIGHT


def bottom_border(width: int) -> str:
    return BOTTOM_LEFT + _repeat(DOUBLE_HORIZONTAL, max(2, width - 2)) + BOTTOM_RIGHT


def separator_line(width: int, *, char: str = HORIZONTAL) -> str:
    return _repeat(char, width)


def box_lines(lines: Iterable[str], *, width: int, title: str = '') -> list[str]:
    content_width = max(1, width - 4)
    rendered = [top_border(width, title=title)]
    for line in lines:
        for wrapped in wrap_line(line, content_width):
            clipped = truncate(wrapped, content_width)
            rendered.append(f'{DOUBLE_VERTICAL} {clipped.ljust(content_width)} {DOUBLE_VERTICAL}')
    rendered.append(bottom_border(width))
    return rendered


def response_lines(value: str, *, width: int) -> list[str]:
    content = value.strip() or 'No response.'
    content_width = max(1, width - 6)
    rendered: list[str] = []
    for raw_line in content.splitlines() or ['']:
        wrapped_lines = wrap_line(raw_line, content_width)
        for index, line in enumerate(wrapped_lines):
            prefix = f'  {RESPONSE_MARKER} ' if not rendered and index == 0 else '    '
            rendered.append(prefix + line)
    return rendered


def status_bar(
    model: str,
    provider: str,
    cwd: str,
    *,
    width: int,
    tokens_used: int = 0,
    max_tokens: int = 0,
    compact_count: int = 0,
) -> str:
    """Build a rich status bar with model info and token usage."""
    left = f' {CIRCLE_FILLED} {model} {DOT_SEPARATOR} {provider}'

    if max_tokens > 0:
        pct = min(100, int(tokens_used * 100 / max_tokens))
        bar_width = 10
        filled = int(pct / 100 * bar_width)
        meter = BLOCK_LOWER * filled + LIGHT_SHADE * (bar_width - filled)
        right = f'{meter} {pct}%'
        if compact_count > 0:
            right += f' {DOT_SEPARATOR} {GEAR}{compact_count}'
    else:
        right = ''

    padding = max(0, width - len(left) - len(right) - 2)
    bar = left + ' ' * padding + right
    return truncate(bar, width)
