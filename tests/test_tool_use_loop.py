from __future__ import annotations

import json
import shutil
import sys
import time
import unittest
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401

from anvil.core.types import StepContext
from anvil.skills import SkillLoader
from anvil.compression import CompressionConfig, TranscriptEntry
from anvil.task_graph import Task, TaskGraph
from anvil.task_store import TaskStore
from anvil.tool_use_loop import ToolUseState, make_tool_use_step
from anvil.todo import TodoItem


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

    def test_should_execute_tool_when_model_adds_text_after_json(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'tool-loop-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            def decider(goal, history, tool_results, state_summary, last_steps) -> str:
                return (
                    '{"thought":"create","plan":["write file"],'
                    '"tool_calls":[{"id":"call_1","name":"write_file",'
                    '"arguments":{"path":"abc/data.json","content":""}}],'
                    '"final":null}'
                    '\nI will create the requested file now.'
                )

            step = make_tool_use_step(decider=decider, workspace_root=tmp_dir)
            context = StepContext(
                goal='create empty json file',
                state=ToolUseState(),
                step_index=0,
                started_at_s=0.0,
                now_s=0.0,
                history=tuple(),
            )

            result = step(context)

            self.assertFalse(result.done)
            self.assertTrue((tmp_dir / 'abc' / 'data.json').is_file())
            self.assertEqual((tmp_dir / 'abc' / 'data.json').read_text(encoding='utf-8'), '')
            self.assertTrue(result.metadata.get('has_tool_calls'))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_not_finish_file_action_without_tool_calls(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'tool-loop-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            def decider(goal, history, tool_results, state_summary, last_steps) -> str:
                return '{"thought":"","plan":[],"tool_calls":[],"final":"done"}'

            step = make_tool_use_step(decider=decider, workspace_root=tmp_dir)
            context = StepContext(
                goal='在D:\\workspace新增一个abc，并在abc新增一个空白的json文件',
                state=ToolUseState(),
                step_index=0,
                started_at_s=0.0,
                now_s=0.0,
                history=tuple(),
            )

            result = step(context)

            self.assertFalse(result.done)
            self.assertEqual(result.metadata.get('missing_tool_calls'), True)
            self.assertIn('tool action required', result.output)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_not_finish_file_action_with_no_tools_and_null_final(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'tool-loop-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            def decider(goal, history, tool_results, state_summary, last_steps) -> str:
                return '{"thought":"","plan":[],"tool_calls":[],"final":null}'

            step = make_tool_use_step(decider=decider, workspace_root=tmp_dir)
            context = StepContext(
                goal='create an empty json file',
                state=ToolUseState(),
                step_index=0,
                started_at_s=0.0,
                now_s=0.0,
                history=tuple(),
            )

            result = step(context)

            self.assertFalse(result.done)
            self.assertEqual(result.metadata.get('missing_tool_calls'), True)
            self.assertIn('tool action required', result.output)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_not_finish_after_unresolved_tool_error(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'tool-loop-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            responses = iter(
                [
                    '{"thought":"try write","plan":["write"],'
                    '"tool_calls":[{"id":"call_1","name":"write_file",'
                    '"arguments":{"path":"..\\\\abc\\\\data.json","content":""}}],'
                    '"final":null}',
                    '{"thought":"","plan":[],"tool_calls":[],"final":"done"}',
                ]
            )

            def decider(goal, history, tool_results, state_summary, last_steps) -> str:
                return next(responses)

            step = make_tool_use_step(decider=decider, workspace_root=tmp_dir)
            first = step(
                StepContext(
                    goal='create json file',
                    state=ToolUseState(),
                    step_index=0,
                    started_at_s=0.0,
                    now_s=0.0,
                    history=tuple(),
                )
            )
            second = step(
                StepContext(
                    goal='create json file',
                    state=first.state,
                    step_index=1,
                    started_at_s=0.0,
                    now_s=0.0,
                    history=(first.output,),
                )
            )

            self.assertFalse(second.done)
            self.assertEqual(second.metadata.get('unresolved_tool_error'), True)
            self.assertIn('tool action failed:', second.output)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_finish_file_action_after_successful_write_tool(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'tool-loop-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            responses = iter(
                [
                    json.dumps(
                        {
                            'thought': 'write requested file',
                            'plan': ['write json'],
                            'tool_calls': [
                                {
                                    'id': 'call_1',
                                    'name': 'write_file',
                                    'arguments': {'path': 'abc/current-path.json', 'content': str(tmp_dir)},
                                }
                            ],
                            'final': None,
                        }
                    ),
                    '{"thought":"done","plan":[],"tool_calls":[],"final":"done"}',
                ]
            )

            def decider(goal, history, tool_results, state_summary, last_steps) -> str:
                return next(responses)

            step = make_tool_use_step(decider=decider, workspace_root=tmp_dir)
            first = step(
                StepContext(
                    goal='create an empty json file containing current path',
                    state=ToolUseState(),
                    step_index=0,
                    started_at_s=0.0,
                    now_s=0.0,
                    history=tuple(),
                )
            )
            second = step(
                StepContext(
                    goal='create an empty json file containing current path',
                    state=first.state,
                    step_index=1,
                    started_at_s=0.0,
                    now_s=0.0,
                    history=(first.output,),
                )
            )

            self.assertTrue(second.done)
            self.assertEqual(second.output, 'done')
            self.assertTrue((tmp_dir / 'abc' / 'current-path.json').is_file())
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

    def test_should_inject_workspace_root_into_state_summary(self) -> None:
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
                state=ToolUseState(),
                step_index=0,
                started_at_s=0.0,
                now_s=0.0,
                history=tuple(),
            )

            step(context)

            self.assertEqual(captured['state_summary']['workspace']['root'], str(tmp_dir))
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
        command = [sys.executable, '-c', 'print("async-ok")']
        try:
            def decider(goal, history, tool_results, state_summary, last_steps) -> str:
                rounds['count'] += 1
                captured['summary'] = state_summary
                captured['tool_results'] = tool_results
                if rounds['count'] == 1:
                    return json.dumps(
                        {
                            'thought': 'launch async',
                            'plan': ['run'],
                            'tool_calls': [
                                {
                                    'id': 'call_async',
                                    'name': 'run_command_async',
                                    'arguments': {'cmd': command},
                                }
                            ],
                            'final': 'later',
                        }
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
