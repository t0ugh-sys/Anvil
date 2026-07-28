"""Integration: full json_loop agent run with MockProvider."""
from __future__ import annotations

import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import _bootstrap  # noqa: F401

import unittest

from anvil.core.agent import AnvilAgent
from anvil.core.types import StopConfig, StopReason
from anvil.llm.mock import _mock_invoke_factory
from anvil.steps.json_loop import JsonLoopState, make_json_decision_step


class TestFullLoop(unittest.TestCase):
    def test_agent_reaches_done_in_two_steps(self):
        invoke = _mock_invoke_factory(model='mock', mode='default')
        step = make_json_decision_step(invoke)
        agent = AnvilAgent(step=step, stop=StopConfig(max_steps=10))
        result = agent.run(goal='test goal', initial_state=JsonLoopState())
        self.assertTrue(result.done)
        self.assertEqual(result.stop_reason, StopReason.done)
        self.assertGreaterEqual(result.steps, 2)

    def test_agent_stops_at_max_steps(self):
        def never_done(_: str) -> str:
            return json.dumps({'answer': 'still going', 'done': False})
        step = make_json_decision_step(never_done)
        agent = AnvilAgent(step=step, stop=StopConfig(max_steps=3))
        result = agent.run(goal='infinite', initial_state=JsonLoopState())
        self.assertFalse(result.done)
        self.assertEqual(result.stop_reason, StopReason.max_steps)
        self.assertEqual(result.steps, 3)

    def test_observer_receives_events(self):
        events: list[str] = []
        def capture(event_type: str, data) -> None:
            events.append(event_type)

        invoke = _mock_invoke_factory(model='mock', mode='default')
        step = make_json_decision_step(invoke)
        agent = AnvilAgent(step=step, stop=StopConfig(max_steps=10))
        agent.run(goal='observe me', initial_state=JsonLoopState(), observer=capture)
        self.assertIn('step_succeeded', events)

    def test_final_output_contains_answer(self):
        invoke = _mock_invoke_factory(model='mock', mode='default')
        step = make_json_decision_step(invoke)
        agent = AnvilAgent(step=step, stop=StopConfig(max_steps=10))
        result = agent.run(goal='what is the answer', initial_state=JsonLoopState())
        self.assertIsNotNone(result.final_output)
        self.assertIsInstance(result.state, JsonLoopState)
        self.assertNotEqual(result.state.last_answer, '')


if __name__ == '__main__':
    unittest.main()
