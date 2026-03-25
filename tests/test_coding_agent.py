from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401

from loop_agent.coding_agent import CodingAgentState, build_coding_step
from loop_agent.core.types import StepContext


class CodingAgentTests(unittest.TestCase):
    def test_should_finish_when_model_stops_calling_tools(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'coding-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            def decider(goal, history, tool_results, state_summary, last_steps) -> str:
                return '{"thought":"all done","plan":[],"tool_calls":[],"final":null}'

            step = build_coding_step(decider, workspace_root=tmp_dir)
            context = StepContext(
                goal='x',
                state=CodingAgentState(),
                step_index=0,
                started_at_s=0.0,
                now_s=0.0,
                history=tuple(),
            )

            result = step(context)

            self.assertTrue(result.done)
            self.assertEqual(result.output, 'all done')
            self.assertEqual(result.metadata.get('has_tool_calls'), False)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_continue_when_model_still_calls_tools(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'coding-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            target = tmp_dir / 'README.md'
            target.write_text('hello', encoding='utf-8')

            def decider(goal, history, tool_results, state_summary, last_steps) -> str:
                return (
                    '{"thought":"read file first","plan":["inspect"],'
                    '"tool_calls":[{"id":"call_1","name":"read_file","arguments":{"path":"README.md"}}],'
                    '"final":"done"}'
                )

            step = build_coding_step(decider, workspace_root=tmp_dir)
            context = StepContext(
                goal='x',
                state=CodingAgentState(),
                step_index=0,
                started_at_s=0.0,
                now_s=0.0,
                history=tuple(),
            )

            result = step(context)

            self.assertFalse(result.done)
            self.assertEqual(result.output, 'continue')
            self.assertEqual(result.metadata.get('has_tool_calls'), True)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
