"""Run all benchmarks and optionally write results to JSON.

Usage:
    python -m tests.benchmarks.run_all
    python -m tests.benchmarks.run_all --output bench_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.benchmarks import bench_tool_loop, bench_compression, bench_token_estimation


_SUITES = [
    ('tool_loop', bench_tool_loop),
    ('compression', bench_compression),
    ('token_estimation', bench_token_estimation),
]


def run_suite(name: str, module) -> dict:
    results: dict[str, float] = {}
    fns = {k: v for k, v in vars(module).items() if k.startswith('bench_')}
    for fn_name, fn in sorted(fns.items()):
        try:
            ns = fn() * 1e9
            results[fn_name] = round(ns, 3)
        except Exception as exc:
            results[fn_name] = f'ERROR: {exc}'
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', metavar='FILE', help='Write JSON results to FILE')
    args = parser.parse_args(argv)

    all_results: dict = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'suites': {},
    }

    for name, module in _SUITES:
        print(f'\n=== {name} ===')
        suite_results = run_suite(name, module)
        all_results['suites'][name] = suite_results
        for fn_name, ns in suite_results.items():
            if isinstance(ns, str):
                print(f'  {fn_name:<55} {ns}')
            elif ns < 1_000:
                print(f'  {fn_name:<55} {ns:8.1f} ns')
            elif ns < 1_000_000:
                print(f'  {fn_name:<55} {ns / 1_000:8.2f} us')
            else:
                print(f'  {fn_name:<55} {ns / 1_000_000:8.3f} ms')

    if args.output:
        Path(args.output).write_text(json.dumps(all_results, indent=2), encoding='utf-8')
        print(f'\nResults written to {args.output}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
