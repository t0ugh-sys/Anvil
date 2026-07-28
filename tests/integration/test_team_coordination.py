"""Integration: two agents chained — first agent's answer feeds second agent's goal."""
from __future__ import annotations

import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import _bootstrap  # noqa: F401

import unittest

from anvil.core.agent import AnvilAgent
from anvil.core.types import StopConfig, StopReason
from anvil.steps.json_loop import JsonLoopState, make_json_decision_step


def _make_invoke(answers: list[str]) -> object:
    """Return an invoke function that cycles through fixed answers."""
    queue = list(answers)

    def invoke(_: str) -> str:
        if len(queue) == 1:
            return json.dumps({'answer': queue[0], 'done': True})
        return json.dumps({'answer': queue.pop(0), 'done': False})

    return invoke


class TestTwoAgentCoordination(unittest.TestCase):
    def test_planner_then_executor(self):
        """Planner produces a plan; executor acts on it."""
        planner_invoke = _make_invoke(['step 1 then step 2', 'refined plan: do A then B'])
        planner_step = make_json_decision_step(planner_invoke)
        planner = AnvilAgent(step=planner_step, stop=StopConfig(max_steps=5))
        plan_result = planner.run(goal='plan how to refactor auth', initial_state=JsonLoopState())
        self.assertTrue(plan_result.done)
        plan = plan_result.state.last_answer
        self.assertNotEqual(plan, '')

        executor_invoke = _make_invoke(['executing step A', 'all steps complete'])
        executor_step = make_json_decision_step(executor_invoke)
        executor = AnvilAgent(step=executor_step, stop=StopConfig(max_steps=5))
        exec_result = executor.run(
            goal=f'Execute this plan: {plan}',
            initial_state=JsonLoopState(),
        )
        self.assertTrue(exec_result.done)
        self.assertNotEqual(exec_result.state.last_answer, '')

    def test_agents_run_independently(self):
        """Two agents with different goals don't share state."""
        invoke_a = _make_invoke(['draft', 'final answer A'])
        invoke_b = _make_invoke(['initial', 'final answer B'])
        step_a = make_json_decision_step(invoke_a)
        step_b = make_json_decision_step(invoke_b)
        agent_a = AnvilAgent(step=step_a, stop=StopConfig(max_steps=5))
        agent_b = AnvilAgent(step=step_b, stop=StopConfig(max_steps=5))

        result_a = agent_a.run(goal='goal A', initial_state=JsonLoopState())
        result_b = agent_b.run(goal='goal B', initial_state=JsonLoopState())

        self.assertTrue(result_a.done)
        self.assertTrue(result_b.done)
        self.assertNotEqual(result_a.state.last_answer, result_b.state.last_answer)

    def test_reviewer_validates_worker_output(self):
        """Reviewer agent checks worker output and signals done only if valid."""
        worker_invoke = _make_invoke(['initial draft', 'polished output'])
        worker_step = make_json_decision_step(worker_invoke)
        worker = AnvilAgent(step=worker_step, stop=StopConfig(max_steps=5))
        work = worker.run(goal='write a summary', initial_state=JsonLoopState())

        reviewer_invoke = _make_invoke(['needs revision', 'approved'])
        reviewer_step = make_json_decision_step(reviewer_invoke)
        reviewer = AnvilAgent(step=reviewer_step, stop=StopConfig(max_steps=5))
        review = reviewer.run(
            goal=f'Review this output and approve or reject: {work.state.last_answer}',
            initial_state=JsonLoopState(),
        )
        self.assertTrue(review.done)
        self.assertIn('approved', review.state.last_answer)


if __name__ == '__main__':
    unittest.main()
