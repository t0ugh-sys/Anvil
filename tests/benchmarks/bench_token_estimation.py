"""Benchmarks for token estimation functions."""
from __future__ import annotations

import timeit
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from anvil.token_estimation import (
    estimate_tokens,
    estimate_text_tokens,
    estimate_content_tokens,
    estimate_message_tokens,
    estimate_messages_tokens,
)


_SHORT = 'Hello, world!'
_MEDIUM = 'word ' * 100
_LONG = 'word ' * 1000
_CJK = '这是一段中文文本，用于测试CJK字符的Token估算速度。' * 20
_JSON_STR = '{"key": "value", "number": 42, "nested": {"a": 1, "b": 2}}' * 10

_MSG_SHORT = {'role': 'user', 'content': _SHORT}
_MSG_LONG = {'role': 'user', 'content': _LONG}
_MSG_LIST = [_MSG_SHORT] * 20

_TOOL_USE_CONTENT = [
    {'type': 'text', 'text': 'Hello'},
    {'type': 'tool_use', 'name': 'read_file', 'input': {'path': 'anvil/utils.py'}},
]


def bench_estimate_text_tokens_short(n: int = 20_000) -> float:
    elapsed = timeit.timeit(lambda: estimate_text_tokens(_SHORT), number=n)
    return elapsed / n


def bench_estimate_text_tokens_long(n: int = 2_000) -> float:
    elapsed = timeit.timeit(lambda: estimate_text_tokens(_LONG), number=n)
    return elapsed / n


def bench_estimate_text_tokens_cjk(n: int = 500) -> float:
    elapsed = timeit.timeit(lambda: estimate_text_tokens(_CJK), number=n)
    return elapsed / n


def bench_estimate_text_tokens_json(n: int = 5_000) -> float:
    elapsed = timeit.timeit(lambda: estimate_text_tokens(_JSON_STR, is_json=True), number=n)
    return elapsed / n


def bench_estimate_content_tool_use(n: int = 10_000) -> float:
    elapsed = timeit.timeit(lambda: estimate_content_tokens(_TOOL_USE_CONTENT), number=n)
    return elapsed / n


def bench_estimate_message_short(n: int = 10_000) -> float:
    elapsed = timeit.timeit(lambda: estimate_message_tokens(_MSG_SHORT), number=n)
    return elapsed / n


def bench_estimate_message_long(n: int = 2_000) -> float:
    elapsed = timeit.timeit(lambda: estimate_message_tokens(_MSG_LONG), number=n)
    return elapsed / n


def bench_estimate_messages_20(n: int = 500) -> float:
    elapsed = timeit.timeit(lambda: estimate_messages_tokens(_MSG_LIST), number=n)
    return elapsed / n


def bench_estimate_tokens_sequence(n: int = 5_000) -> float:
    parts = [_SHORT, _MEDIUM, _SHORT]
    elapsed = timeit.timeit(lambda: estimate_tokens(parts), number=n)
    return elapsed / n


def _fmt(label: str, ns: float) -> str:
    if ns < 1_000:
        return f'{label:<60} {ns:8.1f} ns'
    elif ns < 1_000_000:
        return f'{label:<60} {ns / 1_000:8.2f} us'
    else:
        return f'{label:<60} {ns / 1_000_000:8.3f} ms'


def main():
    print('=== Token Estimation Benchmarks ===')
    results = [
        ('estimate_text_tokens - short 13c (n=200 000)', bench_estimate_text_tokens_short() * 1e9),
        ('estimate_text_tokens - long 5 000c (n=10 000)', bench_estimate_text_tokens_long() * 1e9),
        ('estimate_text_tokens - CJK 20x sentence (n=5 000)', bench_estimate_text_tokens_cjk() * 1e9),
        ('estimate_text_tokens - JSON (n=20 000)', bench_estimate_text_tokens_json() * 1e9),
        ('estimate_content_tokens - tool_use block (n=50 000)', bench_estimate_content_tool_use() * 1e9),
        ('estimate_message_tokens - short message (n=100 000)', bench_estimate_message_short() * 1e9),
        ('estimate_message_tokens - long message (n=10 000)', bench_estimate_message_long() * 1e9),
        ('estimate_messages_tokens - 20 messages (n=10 000)', bench_estimate_messages_20() * 1e9),
        ('estimate_tokens - 3-part sequence (n=50 000)', bench_estimate_tokens_sequence() * 1e9),
    ]
    for label, ns in results:
        print(_fmt(label, ns))


if __name__ == '__main__':
    main()
