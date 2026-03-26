from __future__ import annotations

import shutil
import time
import unittest
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401

from loop_agent.core.types import StepContext
from loop_agent.skills import SkillLoader
from loop_agent.compression import CompressionConfig, TranscriptEntry
from loop_agent.task_graph import Task, TaskGraph
from loop_agent.task_store import TaskStore
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

    def test_should_inject_skill_metadata_without_full_body(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'tool-loop-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        captured = {}
        try:
            loader = SkillLoader()
            self.assertTrue(loader.load('files'))

            def decider(goal, history, tool_results, state_summary, last_steps) -> str:
                captured['state_summary'] = state_summary
                return '{"thought":"done now","plan":[],"tool_calls":[],"final":null}'

            step = make_tool_use_step(decider=decider, workspace_root=tmp_dir, skills=loader)
            context = StepContext(
                goal='x',
                state=ToolUseState(),
                step_index=0,
                started_at_s=0.0,
                now_s=0.0,
                history=tuple(),
            )
            step(context)

            summary = captured['state_summary']
            self.assertIn('available_skills', summary)
            self.assertEqual(summary['available_skills'][0]['name'], 'files')
            self.assertNotIn('read, write, patch', str(summary))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_inject_task_state_from_task_store(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'tool-loop-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        captured = {}
        try:
            store = TaskStore(tmp_dir / '.tasks')
            store.save_graph(
                TaskGraph(
                    [
                        Task(id='t1', title='Inspect', goal='inspect repo'),
                        Task(id='t2', title='Patch', goal='patch repo', dependencies=('t1',)),
                    ]
                )
            )

            def decider(goal, history, tool_results, state_summary, last_steps) -> str:
                captured['state_summary'] = state_summary
                return '{"thought":"done now","plan":[],"tool_calls":[],"final":null}'

            step = make_tool_use_step(decider=decider, workspace_root=tmp_dir, task_store=store)
            context = StepContext(
                goal='x',
                state=ToolUseState(),
                step_index=0,
                started_at_s=0.0,
                now_s=0.0,
                history=tuple(),
            )
            step(context)

            summary = captured['state_summary']
            self.assertIn('task_state', summary)
            self.assertEqual(summary['task_state']['counts']['total'], 2)
            self.assertEqual(summary['task_state']['ready'][0]['id'], 't1')
            self.assertEqual(summary['task_state']['pending'][0]['id'], 't2')
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_micro_compact_old_tool_results(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'tool-loop-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / 'README.md').write_text('hello', encoding='utf-8')
        try:
            def decider(goal, history, tool_results, state_summary, last_steps) -> str:
                return (
                    '{"thought":"inspect","plan":["read"],'
                    '"tool_calls":[{"id":"call_4","name":"read_file","arguments":{"path":"README.md"}}],'
                    '"final":"later"}'
                )

            prior_transcript = (
                TranscriptEntry(kind='tool_result', content='tool-output-1', tool_name='read_file', call_id='call_1', ok=True),
                TranscriptEntry(kind='tool_result', content='tool-output-2', tool_name='search', call_id='call_2', ok=True),
                TranscriptEntry(kind='tool_result', content='tool-output-3', tool_name='write_file', call_id='call_3', ok=True),
            )
            step = make_tool_use_step(
                decider=decider,
                workspace_root=tmp_dir,
                compression_config=CompressionConfig(micro_keep_last_results=3, max_context_tokens=50000),
            )
            context = StepContext(
                goal='x',
                state=ToolUseState(transcript=prior_transcript),
                step_index=0,
                started_at_s=0.0,
                now_s=0.0,
                history=tuple(),
            )
            result = step(context)

            tool_entries = [entry for entry in result.state.transcript if entry.kind == 'tool_result']
            self.assertEqual(tool_entries[0].content, '[Previous: used read_file]')
            self.assertEqual(tool_entries[-1].content, 'hello')
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_auto_compact_and_archive_transcript(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'tool-loop-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            def decider(goal, history, tool_results, state_summary, last_steps) -> str:
                return '{"thought":"done now","plan":[],"tool_calls":[],"final":null}'

            transcripts_dir = tmp_dir / '.transcripts'
            step = make_tool_use_step(
                decider=decider,
                workspace_root=tmp_dir,
                compression_config=CompressionConfig(max_context_tokens=5),
                transcripts_dir=transcripts_dir,
                summarizer=lambda goal, previous_summary, transcript: 'compressed summary',
            )
            context = StepContext(
                goal='x',
                state=ToolUseState(
                    transcript=(TranscriptEntry(kind='thought', content='a' * 200),),
                ),
                step_index=0,
                started_at_s=0.0,
                now_s=0.0,
                history=tuple(),
            )
            result = step(context)

            self.assertEqual(result.state.compaction_count, 1)
            self.assertEqual(result.state.compact_summary, 'compressed summary')
            self.assertEqual(result.state.transcript[0].kind, 'summary')
            self.assertTrue((transcripts_dir / 'compact_0001.json').exists())
            self.assertEqual(result.metadata['compression_state']['last_compaction_reason'], 'auto:52>5')
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_allow_manual_compact_tool(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'tool-loop-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            def decider(goal, history, tool_results, state_summary, last_steps) -> str:
                return (
                    '{"thought":"compact now","plan":["compress"],'
                    '"tool_calls":[{"id":"call_1","name":"compact","arguments":{"reason":"manual checkpoint"}}],'
                    '"final":"later"}'
                )

            step = make_tool_use_step(
                decider=decider,
                workspace_root=tmp_dir,
                transcripts_dir=tmp_dir / '.transcripts',
                summarizer=lambda goal, previous_summary, transcript: 'manual summary',
            )
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
            self.assertEqual(result.state.compact_summary, 'manual summary')
            self.assertEqual(result.state.last_compaction_reason, 'manual checkpoint')
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_drain_background_notifications_before_decider(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'tool-loop-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        captured = {}
        rounds = {'count': 0}
        try:
            def decider(goal, history, tool_results, state_summary, last_steps) -> str:
                rounds['count'] += 1
                captured['summary'] = state_summary
                captured['tool_results'] = tool_results
                if rounds['count'] == 1:
                    return (
                        '{"thought":"launch async","plan":["run"],'
                        '"tool_calls":[{"id":"call_async","name":"run_command_async","arguments":{"cmd":["cmd","/c","echo async-ok"]}}],'
                        '"final":"later"}'
                    )
                return '{"thought":"done now","plan":[],"tool_calls":[],"final":null}'

            step = make_tool_use_step(decider=decider, workspace_root=tmp_dir)
            context1 = StepContext(
                goal='x',
                state=ToolUseState(),
                step_index=0,
                started_at_s=0.0,
                now_s=0.0,
                history=tuple(),
            )
            result1 = step(context1)
            self.assertFalse(result1.done)
            self.assertEqual(result1.metadata['tool_calls'][0]['name'], 'run_command_async')

            for _ in range(20):
                time.sleep(0.05)
                context2 = StepContext(
                    goal='x',
                    state=result1.state,
                    step_index=1,
                    started_at_s=0.0,
                    now_s=0.1,
                    history=('continue',),
                )
                result2 = step(context2)
                notifications = result2.metadata.get('background_notifications', [])
                if notifications:
                    self.assertTrue(any('async-ok' in item.get('output', '') for item in notifications))
                    self.assertTrue(any(item.get('id') == 'call_async' for item in notifications))
                    self.assertTrue(captured['summary']['notification_queue'])
                    return

            self.fail('background notification was not delivered in time')
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
