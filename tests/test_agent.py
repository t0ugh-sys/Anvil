from __future__ import annotations

import time
import unittest
from dataclasses import dataclass

import _bootstrap  # noqa: F401

from loop_agent.core.agent import LoopAgent
from loop_agent.core.types import StepContext, StepResult, StopConfig, StopReason


@dataclass(frozen=True)
class _State:
    value: int = 0


class LoopAgentTests(unittest.TestCase):
    def test_should_stop_when_done(self) -> None:
        def step(context: StepContext[_State]) -> StepResult[_State]:
            if context.step_index >= 2:
                return StepResult(output='ok', state=_State(value=3), done=True)
            return StepResult(output=f'step{context.step_index}', state=_State(value=context.step_index + 1))

        agent = LoopAgent(step=step, stop=StopConfig(max_steps=10, max_elapsed_s=10.0))
        result = agent.run(goal='x', initial_state=_State())

        self.assertTrue(result.done)
        self.assertEqual(result.stop_reason, StopReason.done)
        self.assertEqual(result.steps, 3)
        self.assertEqual(result.final_output, 'ok')
        self.assertEqual(result.history, ('step0', 'step1', 'ok'))

    def test_should_stop_on_max_steps(self) -> None:
        def step(context: StepContext[_State]) -> StepResult[_State]:
            return StepResult(output='no', state=context.state, done=False)

        agent = LoopAgent(step=step, stop=StopConfig(max_steps=3, max_elapsed_s=10.0))
        result = agent.run(goal='x', initial_state=_State())

        self.assertFalse(result.done)
        self.assertEqual(result.stop_reason, StopReason.max_steps)
        self.assertEqual(result.steps, 3)
        self.assertEqual(len(result.history), 3)

    def test_should_stop_on_timeout(self) -> None:
        def step(context: StepContext[_State]) -> StepResult[_State]:
            time.sleep(0.02)
            return StepResult(output='slow', state=context.state, done=False)

        agent = LoopAgent(step=step, stop=StopConfig(max_steps=20, max_elapsed_s=0.01))
        result = agent.run(goal='x', initial_state=_State())

        self.assertFalse(result.done)
        self.assertEqual(result.stop_reason, StopReason.timeout)

    def test_should_stop_on_cancelled(self) -> None:
        def step(context: StepContext[_State]) -> StepResult[_State]:
            return StepResult(output='loop', state=context.state, done=False)

        cancel_counter = {'calls': 0}

        def is_cancelled() -> bool:
            cancel_counter['calls'] += 1
            return cancel_counter['calls'] >= 2

        agent = LoopAgent(step=step, stop=StopConfig(max_steps=20, max_elapsed_s=10.0))
        result = agent.run(goal='x', initial_state=_State(), is_cancelled=is_cancelled)

        self.assertFalse(result.done)
        self.assertEqual(result.stop_reason, StopReason.cancelled)
        self.assertEqual(result.steps, 1)

    def test_should_capture_step_error(self) -> None:
        def step(context: StepContext[_State]) -> StepResult[_State]:
            raise RuntimeError('boom')

        agent = LoopAgent(step=step, stop=StopConfig(max_steps=20, max_elapsed_s=10.0))
        result = agent.run(goal='x', initial_state=_State())

        self.assertFalse(result.done)
        self.assertEqual(result.stop_reason, StopReason.step_error)
        self.assertEqual(result.error, 'boom')

    def test_should_raise_for_empty_goal(self) -> None:
        def step(context: StepContext[_State]) -> StepResult[_State]:
            return StepResult(output='x', state=context.state, done=True)

        agent = LoopAgent(step=step)
        with self.assertRaises(ValueError):
            agent.run(goal='   ', initial_state=_State())


if __name__ == '__main__':
    unittest.main()
