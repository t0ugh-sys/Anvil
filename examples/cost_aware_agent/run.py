"""
Cost-Aware Agent — AnvilAgent with hard resource limits.

Demonstrates StopConfig with step and time ceilings to prevent runaway agents.
Replace `mock_invoke` with a real provider and add a token-counting wrapper
to enforce a token budget.

Run:
    python examples/cost_aware_agent/run.py
"""

from __future__ import annotations

import json

from anvil.core.agent import AnvilAgent
from anvil.core.types import StopConfig
from anvil.steps.json_loop import JsonLoopState, make_json_decision_step


class BudgetedInvoke:
    """Wrapper that counts calls and refuses once the budget is exceeded."""

    def __init__(self, max_calls: int) -> None:
        self._max_calls = max_calls
        self._calls = 0

    @property
    def calls_used(self) -> int:
        return self._calls

    def __call__(self, prompt: str) -> str:
        self._calls += 1
        if self._calls >= self._max_calls:
            # Signal the agent to stop immediately
            return json.dumps({'answer': 'Budget reached - stopping early.', 'done': True})
        # Simulate work: each step makes partial progress
        return json.dumps({
            'thought': f'Working (call {self._calls}/{self._max_calls})...',
            'answer': f'Intermediate result {self._calls}',
            'done': False,
        })


def main() -> None:
    budget = BudgetedInvoke(max_calls=3)

    step = make_json_decision_step(budget, history_window=4)
    agent = AnvilAgent(
        step=step,
        stop=StopConfig(
            max_steps=10,        # hard ceiling even if LLM never sets done=True
            max_elapsed_s=30.0,  # wall-clock timeout
        ),
    )

    result = agent.run(
        goal='Analyse a codebase with a strict 3-call budget',
        initial_state=JsonLoopState(),
    )

    print(f'Done:        {result.done}')
    print(f'Stop reason: {result.stop_reason.value}')
    print(f'Steps used:  {result.steps}  (max 10)')
    print(f'LLM calls:   {budget.calls_used}  (budget 3)')
    print(f'Answer:      {result.final_output}')


if __name__ == '__main__':
    main()
