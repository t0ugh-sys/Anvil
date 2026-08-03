"""Benchmarks for context compression strategies."""
from __future__ import annotations

import timeit
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from anvil.compression import (
    estimate_tokens,
    micro_compact_entries,
    summarize_entries_deterministically,
    TranscriptEntry,
    time_based_micro_compact,
)


def _make_entries(n: int) -> tuple:
    entries = []
    for i in range(n):
        kind = 'thought' if i % 2 == 0 else 'tool_result'
        entries.append(TranscriptEntry(kind=kind, content=f'Message {i}: ' + 'word ' * 20))
    return tuple(entries)


def bench_estimate_tokens_short(n: int = 100_000) -> float:
    text = 'Hello, world! This is a short message.'
    elapsed = timeit.timeit(lambda: estimate_tokens([text]), number=n)
    return elapsed / n


def bench_estimate_tokens_long(n: int = 10_000) -> float:
    text = 'word ' * 500
    elapsed = timeit.timeit(lambda: estimate_tokens([text]), number=n)
    return elapsed / n


def bench_micro_compact_10(n: int = 5_000) -> float:
    entries = _make_entries(10)
    elapsed = timeit.timeit(lambda: micro_compact_entries(entries, keep_last_results=3), number=n)
    return elapsed / n


def bench_micro_compact_100(n: int = 500) -> float:
    entries = _make_entries(100)
    elapsed = timeit.timeit(lambda: micro_compact_entries(entries, keep_last_results=3), number=n)
    return elapsed / n


def bench_summarize_deterministic(n: int = 1_000) -> float:
    entries = _make_entries(20)
    goal = 'Refactor the codebase'
    elapsed = timeit.timeit(
        lambda: summarize_entries_deterministically(goal=goal, previous_summary='', entries=entries),
        number=n,
    )
    return elapsed / n


def bench_time_based_micro_compact(n: int = 1_000) -> float:
    entries = _make_entries(50)
    elapsed = timeit.timeit(lambda: time_based_micro_compact(entries), number=n)
    return elapsed / n


def _fmt(label: str, ns: float) -> str:
    if ns < 1_000:
        return f'{label:<55} {ns:8.1f} ns'
    elif ns < 1_000_000:
        return f'{label:<55} {ns / 1_000:8.2f} us'
    else:
        return f'{label:<55} {ns / 1_000_000:8.3f} ms'


def main():
    print('=== Compression Benchmarks ===')
    results = [
        ('estimate_tokens - short text (n=100 000)', bench_estimate_tokens_short() * 1e9),
        ('estimate_tokens - long text 500w (n=10 000)', bench_estimate_tokens_long() * 1e9),
        ('micro_compact_entries - 10 entries (n=5 000)', bench_micro_compact_10() * 1e9),
        ('micro_compact_entries - 100 entries (n=500)', bench_micro_compact_100() * 1e9),
        ('summarize_entries_deterministically - 20 entries (n=1 000)', bench_summarize_deterministic() * 1e9),
        ('time_based_micro_compact - 50 entries (n=1 000)', bench_time_based_micro_compact() * 1e9),
    ]
    for label, ns in results:
        print(_fmt(label, ns))


if __name__ == '__main__':
    main()
