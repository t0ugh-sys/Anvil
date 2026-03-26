from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401

from loop_agent.core.types import StepContext
from loop_agent.tool_use_loop import ToolUseState, make_tool_use_step
from loop_agent.todo import TodoItem


class ToolUseLoopTests(unittest.TestCase):
    def test_should_stop_when_model_has_no_tool_calls(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'tool-loop-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            def decider(goal, history, tool_results, state_summary, last_steps) -> str:
                return '{"thought":"done now","plan":[],"tool_calls":[],"final":null}'

            step = make_tool_use_step(decider=decider, workspace_root=tmp_dir)
            context = StepContext(
                goal='x',
                state=ToolUseState(),
                step_index=0,
                started_at_s=0.0,
                now_s=0.0,
                history=tuple(),
            )
            result = step(context)
            self.assertTrue(result.done)
            self.assertEqual(result.output, 'done now')
            self.assertFalse(result.metadata.get('has_tool_calls'))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_execute_tool_and_continue(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'tool-loop-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        target = tmp_dir / 'README.md'
        target.write_text('hello', encoding='utf-8')
        try:
            def decider(goal, history, tool_results, state_summary, last_steps) -> str:
                return (
                    '{"thought":"inspect","plan":["read"],'
                    '"tool_calls":[{"id":"call_1","name":"read_file","arguments":{"path":"README.md"}}],'
                    '"final":"later"}'
                )

            step = make_tool_use_step(decider=decider, workspace_root=tmp_dir)
            context = StepContext(
                goal='x',
                state=ToolUseState(),
                step_index=0,
                started_at_s=0.0,
                now_s=0.0,
                history=tuple(),
            )
            result = step(context)
            self.assertFalse(result.done)
            self.assertEqual(result.output, 'continue')
            self.assertTrue(result.metadata.get('has_tool_calls'))
            tool_results = result.metadata.get('tool_results', [])
            self.assertEqual(tool_results[0]['ok'], True)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_append_thought_and_tool_history(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'tool-loop-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        target = tmp_dir / 'README.md'
        target.write_text('hello', encoding='utf-8')
        try:
            def decider(goal, history, tool_results, state_summary, last_steps) -> str:
                return (
                    '{"thought":"inspect readme","plan":["read"],'
                    '"tool_calls":[{"id":"call_1","name":"read_file","arguments":{"path":"README.md"}}],'
                    '"final":"later"}'
                )

            step = make_tool_use_step(decider=decider, workspace_root=tmp_dir)
            context = StepContext(
                goal='x',
                state=ToolUseState(history=('existing',)),
                step_index=0,
                started_at_s=0.0,
                now_s=0.0,
                history=tuple(),
            )
            result = step(context)

            self.assertEqual(result.state.history[0], 'existing')
            self.assertIn('thought: inspect readme', result.state.history)
            self.assertIn('tool[call_1] ok', result.state.history)
            self.assertEqual(result.metadata.get('tool_calls')[0]['name'], 'read_file')
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_update_todo_state_via_tool(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'tool-loop-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            def decider(goal, history, tool_results, state_summary, last_steps) -> str:
                return (
                    '{"thought":"update todos","plan":["track"],'
                    '"tool_calls":[{"id":"call_1","name":"todo_write","arguments":{"items":['
                    '{"id":"t1","content":"inspect repo","status":"completed"},'
                    '{"id":"t2","content":"edit file","status":"in_progress"}]}}],'
                    '"final":"later"}'
                )

            step = make_tool_use_step(decider=decider, workspace_root=tmp_dir)
            context = StepContext(
                goal='x',
                state=ToolUseState(rounds_since_todo_update=4),
                step_index=0,
                started_at_s=0.0,
                now_s=0.0,
                history=tuple(),
            )
            result = step(context)

            self.assertFalse(result.done)
            self.assertEqual(result.state.rounds_since_todo_update, 0)
            self.assertEqual(result.state.todos[0].id, 't1')
            self.assertEqual(result.state.todos[1].status, 'in_progress')
            self.assertIn('todo_state', result.metadata)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_inject_todo_reminder_after_stale_rounds(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'tool-loop-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        captured = {}
        try:
            def decider(goal, history, tool_results, state_summary, last_steps) -> str:
                captured['state_summary'] = state_summary
                return '{"thought":"done now","plan":[],"tool_calls":[],"final":null}'

            step = make_tool_use_step(decider=decider, workspace_root=tmp_dir)
            context = StepContext(
                goal='x',
                state=ToolUseState(
                    todos=(TodoItem(id='t1', content='keep visible progress', status='in_progress'),),
                    rounds_since_todo_update=3,
                ),
                step_index=0,
                started_at_s=0.0,
                now_s=0.0,
                history=tuple(),
            )
            step(context)

            summary = captured['state_summary']
            self.assertIn('todo_state', summary)
            self.assertIn('todo_reminder', summary)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
