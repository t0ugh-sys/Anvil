"""
Multi-Agent Team — two AnvilAgent instances collaborating on a task.

A Planner agent breaks the task into steps; an Executor agent carries each step out.
Both use mock LLM invokes. Replace with real providers to run against Claude.

Run:
    python examples/multi_agent_team/run.py
"""

from __future__ import annotations

import json

from anvil.core.agent import AnvilAgent
from anvil.core.types import StopConfig
from anvil.steps.json_loop import JsonLoopState, make_json_decision_step


# ---------------------------------------------------------------------------
# Mock LLM responses

def planner_invoke(prompt: str) -> str:
    """Mock planner — decomposes a goal into a list of steps."""
    return json.dumps({
        'thought': 'Breaking the refactoring task into atomic steps.',
        'answer': json.dumps([
            'Rename `calc` to `calculate`',
            'Extract `validate_input` helper',
            'Add type annotations',
        ]),
        'done': True,
    })


def executor_invoke_for(step: str):
    """Returns a mock executor invoke that reports completing one step."""
    def _invoke(prompt: str) -> str:
        return json.dumps({
            'thought': f'Executing: {step}',
            'answer': f'Done: {step}',
            'done': True,
        })
    return _invoke


# ---------------------------------------------------------------------------

def run_planner(goal: str) -> list[str]:
    """Run the Planner agent; returns a list of sub-tasks."""
    step = make_json_decision_step(planner_invoke, history_window=2)
    agent = AnvilAgent(step=step, stop=StopConfig(max_steps=3, max_elapsed_s=10.0))
    result = agent.run(goal=goal, initial_state=JsonLoopState())

    try:
        return json.loads(result.final_output or '[]')
    except (json.JSONDecodeError, TypeError):
        return [result.final_output or goal]


def run_executor(sub_task: str) -> str:
    """Run the Executor agent on a single sub-task; returns output."""
    invoke = executor_invoke_for(sub_task)
    step = make_json_decision_step(invoke, history_window=2)
    agent = AnvilAgent(step=step, stop=StopConfig(max_steps=3, max_elapsed_s=10.0))
    result = agent.run(goal=sub_task, initial_state=JsonLoopState())
    return result.final_output or ''


def main() -> None:
    goal = 'Refactor the legacy calculator module for readability and type safety'

    print(f'Goal: {goal}')
    print()

    # Phase 1 — Planning
    print('[Planner] Decomposing goal into steps...')
    sub_tasks = run_planner(goal)
    print(f'  Plan ({len(sub_tasks)} steps):')
    for i, task in enumerate(sub_tasks, 1):
        print(f'    {i}. {task}')
    print()

    # Phase 2 — Execution
    print('[Executor] Running each step...')
    outputs = []
    for task in sub_tasks:
        output = run_executor(task)
        outputs.append(output)
        print(f'  + {output}')

    print()
    print(f'Team complete. {len(outputs)}/{len(sub_tasks)} steps executed.')


if __name__ == '__main__':
    main()
