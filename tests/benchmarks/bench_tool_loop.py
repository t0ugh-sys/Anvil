"""Benchmarks for the tool execution loop."""
from __future__ import annotations

import timeit
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from anvil.agent.loop import ToolUseState, make_tool_use_step
from anvil.infra.policies import ToolPolicy


def _make_mock_decider(response: str):
    def decider(goal, history, tool_results, state_summary, last_steps):
        return response
    return decider


FINAL_RESPONSE = '{"thought":"done","tool_calls":[],"final":"Task complete"}'


def bench_empty_step_construction(n: int = 1000) -> float:
    """Time to construct a tool-use step closure."""
    workspace = Path('.')
    decider = _make_mock_decider(FINAL_RESPONSE)

    def _run():
        make_tool_use_step(
            decider=decider,
            workspace_root=workspace,
            policy=ToolPolicy.allow_all(),
        )

    elapsed = timeit.timeit(_run, number=n)
    return elapsed / n


def bench_tool_use_state_creation(n: int = 50_000) -> float:
    """Time to instantiate a fresh ToolUseState."""
    elapsed = timeit.timeit(ToolUseState, number=n)
    return elapsed / n


def bench_policy_allow_all(n: int = 100_000) -> float:
    """Time to create and query allow_all policy."""
    def _run():
        policy = ToolPolicy.allow_all()
        policy.allows_tool('read_file')

    elapsed = timeit.timeit(_run, number=n)
    return elapsed / n


def _fmt(label: str, ns: float) -> str:
    if ns < 1_000:
        return f'{label:<45} {ns:8.1f} ns'
    elif ns < 1_000_000:
        return f'{label:<45} {ns / 1_000:8.2f} us'
    else:
        return f'{label:<45} {ns / 1_000_000:8.3f} ms'


def main():
    print('=== Tool Loop Benchmarks ===')
    results = [
        ('ToolUseState() instantiation (n=50 000)', bench_tool_use_state_creation() * 1e9),
        ('ToolPolicy.allow_all() + is_allowed (n=100 000)', bench_policy_allow_all() * 1e9),
        ('make_tool_use_step() construction (n=1 000)', bench_empty_step_construction() * 1e9),
    ]
    for label, ns in results:
        print(_fmt(label, ns))


if __name__ == '__main__':
    main()
