from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from loop_agent.core.types import StepContext
from loop_agent.steps.json_loop import JsonLoopState, make_json_decision_step


class JsonLoopStepTests(unittest.TestCase):
    def test_should_return_done_when_invoke_returns_done_json(self) -> None:
        def invoke(_: str) -> str:
            return '{"answer":"ok","done":true}'

        step = make_json_decision_step(invoke, history_window=2)
        context = StepContext(
            goal='x',
            state=JsonLoopState(),
            step_index=0,
            started_at_s=0.0,
            now_s=0.0,
            history=tuple(),
        )
        result = step(context)
        self.assertTrue(result.done)
        self.assertEqual(result.output, 'ok')
        self.assertEqual(result.state.last_answer, 'ok')

    def test_should_mark_parse_error_when_output_is_not_json(self) -> None:
        def invoke(_: str) -> str:
            return 'not-json'

        step = make_json_decision_step(invoke)
        context = StepContext(
            goal='x',
            state=JsonLoopState(),
            step_index=0,
            started_at_s=0.0,
            now_s=0.0,
            history=tuple(),
        )
        result = step(context)
        self.assertFalse(result.done)
        self.assertEqual(result.metadata.get('parse_error'), True)
        self.assertEqual(result.output, 'not-json')

    def test_should_raise_when_history_window_is_negative(self) -> None:
        def invoke(_: str) -> str:
            return '{"answer":"x","done":false}'

        with self.assertRaises(ValueError):
            make_json_decision_step(invoke, history_window=-1)


if __name__ == '__main__':
    unittest.main()

