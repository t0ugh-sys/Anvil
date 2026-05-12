from __future__ import annotations

import textwrap
from typing import Iterable


MIN_UI_WIDTH = 48
MAX_UI_WIDTH = 88

TOP_LEFT = '\u256d'
TOP_RIGHT = '\u256e'
BOTTOM_LEFT = '\u2570'
BOTTOM_RIGHT = '\u256f'
HORIZONTAL = '\u2500'
VERTICAL = '\u2502'
RESPONSE_MARKER = '\u23bf'
WORKING_MARKER = '\u273b'
DOT_SEPARATOR = '\u00b7'
PROMPT_MARKER = '\u276f'

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
    return value[: width - 3] + '...'


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


def top_border(width: int, *, title: str = '') -> str:
    inner_width = max(2, width - 2)
    if not title:
        return TOP_LEFT + (HORIZONTAL * inner_width) + TOP_RIGHT
    label = truncate(f' {title} ', inner_width)
    return TOP_LEFT + label + (HORIZONTAL * max(0, inner_width - len(label))) + TOP_RIGHT


def bottom_border(width: int) -> str:
    return BOTTOM_LEFT + (HORIZONTAL * max(2, width - 2)) + BOTTOM_RIGHT


def box_lines(lines: Iterable[str], *, width: int, title: str = '') -> list[str]:
    content_width = max(1, width - 4)
    rendered = [top_border(width, title=title)]
    for line in lines:
        for wrapped in wrap_line(line, content_width):
            clipped = truncate(wrapped, content_width)
            rendered.append(f'{VERTICAL} {clipped.ljust(content_width)} {VERTICAL}')
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
