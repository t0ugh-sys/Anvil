# -*- coding: utf-8 -*-
from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401

from anvil.commands import (
    execute_slash_command,
    format_event_summary,
    format_history_summary,
    format_permission_summary,
    format_session_panel,
    format_summary_text,
    format_status_summary,
    format_todo_summary,
    parse_slash_command,
)
from anvil.services.event_viewer import render_event_row
from anvil.services.coding_runtime import build_coding_decider, build_coding_prompt
from anvil.services.session_runtime import (
    _extract_interactive_output,
    _format_chat_history,
    _looks_like_action_request,
    _should_use_plain_chat_fallback,
    build_interactive_turn_runner,
    build_interactive_parser,
)
from anvil.agent.protocol import ToolResult
from anvil.runtime.session import SessionStore
from anvil.infra.skills import SkillLoader
from anvil.tools import builtin_tool_specs


class AgentCliTests(unittest.TestCase):
    def test_should_only_include_skill_metadata_in_prompt(self) -> None:
        loader = SkillLoader()
        self.assertTrue(loader.load('files'))
        captured = {}

        def fake_invoke(prompt: str) -> str:
            captured['prompt'] = prompt
            return '{"thought":"done","plan":[],"tool_calls":[],"final":"done"}'

        args = build_interactive_parser().parse_args(['--provider', 'mock', '--model', 'mock-v3'])

        with patch('anvil.services.coding_runtime.build_invoke_from_args', return_value=fake_invoke):
            decider = build_coding_decider(args, loader)
            decider('goal', tuple(), tuple(), {}, tuple())

        prompt = captured['prompt']
        self.assertIn('Available skills:', prompt)
        self.assertIn('- files: Read, write, patch, and search files', prompt)
        self.assertNotIn('# Anvil Skills', prompt)

    def test_should_instruct_coding_model_to_use_tools_for_file_operations(self) -> None:
        captured = {}

        def fake_invoke(prompt: str) -> str:
            captured['prompt'] = prompt
            return '{"thought":"done","plan":[],"tool_calls":[],"final":"done"}'

        args = build_interactive_parser().parse_args(['--provider', 'mock', '--model', 'mock-v3'])

        with patch('anvil.services.coding_runtime.build_invoke_from_args', return_value=fake_invoke):
            decider = build_coding_decider(args, None)
            decider('create an empty markdown file', tuple(), tuple(), {}, tuple())

        prompt = captured['prompt']
        self.assertIn('do the work with tools', prompt)
        self.assertIn('For creating an empty file, use write_file', prompt)
        self.assertIn('Do not answer with shell commands', prompt)
        self.assertIn('field types, not values to copy', prompt)
        self.assertIn('- write_file:', prompt)
        self.assertNotIn('"name":"read_file","arguments":{"path":"README.md"}', prompt)
        self.assertNotIn('"name":"tool_name"', prompt)
        self.assertNotIn('short private status', prompt)
        self.assertNotIn('short next step', prompt)

    def test_should_repair_prompt_after_invalid_agent_step_json(self) -> None:
        prompt = build_coding_prompt(
            goal='create an empty json file',
            history=('invalid agent step json. expected schema: {}',),
            tool_results=tuple(),
            state_summary={},
            last_steps=tuple(),
            history_window=8,
        )

        self.assertIn('previous response did not match', prompt)
        self.assertIn('Return exactly one JSON object', prompt)
        self.assertIn('Use write_file for empty files', prompt)

    def test_should_repair_prompt_after_missing_tool_calls_for_file_action(self) -> None:
        prompt = build_coding_prompt(
            goal='create an empty json file',
            history=tuple(),
            tool_results=tuple(),
            state_summary={'workspace': {'root': 'D:\\workspace\\Anvil'}},
            last_steps=('tool action required: file operation requests must be completed with tool calls',),
            history_window=8,
        )

        self.assertIn('tool_calls was empty for a file operation', prompt)
        self.assertIn('at least one tool call', prompt)
        self.assertIn('Do not set final until a tool result confirms', prompt)

    def test_should_prompt_for_final_after_successful_tool_result(self) -> None:
        prompt = build_coding_prompt(
            goal='create an empty json file',
            history=tuple(),
            tool_results=(ToolResult(id='call_1', ok=True, output='ok'),),
            state_summary={'workspace': {'root': 'D:\\workspace\\Anvil'}},
            last_steps=('continue',),
            history_window=8,
        )

        self.assertIn('previous tool call succeeded', prompt)
        self.assertIn('do not call more tools', prompt)
        self.assertIn('return final', prompt)

    def test_should_extract_interactive_output_from_fallback_fields(self) -> None:
        self.assertEqual(_extract_interactive_output({'final_output': 'done'}), 'done')
        self.assertEqual(_extract_interactive_output({'history': ['', 'last visible']}), 'last visible')
        self.assertEqual(_extract_interactive_output({'error': 'bad json'}), 'Run failed: bad json')
        self.assertEqual(
            _extract_interactive_output({'stop_reason': 'max_steps'}),
            'Stopped without final output (reason: max_steps).',
        )

    def test_should_use_plain_chat_fallback_for_provider_format_errors(self) -> None:
        self.assertTrue(_should_use_plain_chat_fallback('Stopped without final output (reason: max_steps).'))
        self.assertTrue(_should_use_plain_chat_fallback('invalid agent step json. expected schema: {}'))
        self.assertTrue(_should_use_plain_chat_fallback('Run failed: invalid Anthropic response format'))
        self.assertTrue(_should_use_plain_chat_fallback('done'))
        self.assertTrue(_should_use_plain_chat_fallback('ok'))
        self.assertTrue(_should_use_plain_chat_fallback(''))
        self.assertFalse(_should_use_plain_chat_fallback('I am Anvil, your coding assistant.'))

    def test_should_detect_action_requests_that_must_not_plain_chat_fallback(self) -> None:
        self.assertTrue(_looks_like_action_request('在D:\\workspace新增一个abc，并在abc新建一个.md文件'))
        self.assertTrue(_looks_like_action_request('create a file at notes/todo.md'))
        self.assertFalse(_looks_like_action_request('你是谁'))

    def test_should_format_chat_history_without_current_user_message(self) -> None:
        history = _format_chat_history(
            [
                'user: 你是谁',
                'assistant: 我是 Anvil',
                'user: 把对话内容写到txt中',
                'assistant: 1. 保存目前的对话\n2. 继续对话后再保存',
                'user: 1',
            ],
            current_user_text='1',
        )
        self.assertIn('assistant: 1. 保存目前的对话', history)
        self.assertNotIn('\nuser: 1', history)

    def test_interactive_turn_runner_marks_trusted_workspace(self) -> None:
        args = build_interactive_parser().parse_args(['--provider', 'mock', '--model', 'mock-v3'])
        captured = {}

        class FakeRuntime:
            def __init__(self, runtime_args, *, goal: str) -> None:
                captured['trusted'] = getattr(runtime_args, 'interactive_trusted_workspace', False)
                captured['goal'] = goal
                self.goal = goal
                self.workspace_root = Path('.')
                self.observer = None
                self.session_store = SessionStore.create(
                    root_dir=Path('tests/.tmp') / f'interactive-{uuid.uuid4().hex}' / 'sessions',
                    workspace_root=Path('.'),
                    goal=goal,
                    memory_run_dir=Path('tests/.tmp') / 'runs',
                )
                self.task_store = None
                self.compression_config = None
                self.transcripts_dir = None

            def build_context_provider(self):
                return None

            def build_policy(self):
                return None

            def finalize(self, result):
                return {'final_output': 'done'}

        with patch('anvil.services.session_runtime.CodeRuntime', FakeRuntime):
            with patch('anvil.services.session_runtime.load_skills_from_args', return_value=None):
                with patch('anvil.services.session_runtime.build_coding_decider', return_value=lambda *args: ''):
                    with patch('anvil.services.session_runtime.build_coding_summarizer', return_value=None):
                        with patch('anvil.services.session_runtime.run_coding_agent') as run_agent:
                            run_agent.return_value = type(
                                'Result',
                                (),
                                {'done': True, 'stop_reason': type('Stop', (), {'value': 'done'})(), 'steps': 1},
                            )()
                            from anvil.llm.usage import TokenUsageTracker
                            from anvil.llm.rate_limit import RateLimitTracker
                            runner = build_interactive_turn_runner(
                                args,
                                session_id='s1',
                                usage_tracker=TokenUsageTracker(),
                                rate_limit_tracker=RateLimitTracker(),
                            )
                            output = runner('create file')

        self.assertEqual(output, 'done')
        self.assertEqual(captured['trusted'], True)
        self.assertEqual(captured['goal'], 'create file')

    def test_should_parse_help_and_resume_slash_commands(self) -> None:
        self.assertEqual(parse_slash_command('/help').name, 'help')
        self.assertEqual(parse_slash_command('/resume now').argument, 'now')
        self.assertEqual(parse_slash_command('/status').name, 'status')
        self.assertEqual(parse_slash_command('/history 12').argument, '12')
        self.assertEqual(parse_slash_command('/panel').name, 'panel')
        self.assertIsNone(parse_slash_command('plain text'))

    def test_should_format_session_views(self) -> None:
        tmp_dir = Path('D:/workspace/Anvil/.tmp') / f'session-view-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            session_store = SessionStore.create(
                root_dir=tmp_dir,
                workspace_root=tmp_dir,
                goal='inspect runtime',
                memory_run_dir=tmp_dir / 'memory',
            )
            session_store.state.last_summary = 'repo inspected'
            session_store.state.permission_stats = {'allow': 2, 'deny': 1, 'ask': 3}
            session_store.state.permission_cache = {'read_file:*': 'allow'}
            session_store.state.todo_state = {
                'items': [
                    {'content': 'inspect repo', 'status': 'completed'},
                    {'content': 'edit runtime', 'status': 'in_progress'},
                ]
            }
            session_store.append_event('chat_user', {'role': 'user', 'content': 'hello'})
            session_store.append_event('chat_assistant', {'role': 'assistant', 'content': 'hi'})
            self.assertIn('session_id:', format_status_summary(session_store))
            self.assertIn('recent_history:', format_history_summary(session_store))
            self.assertIn('summary:\nrepo inspected', format_summary_text(session_store))
            self.assertIn('recent_events:', format_event_summary(session_store))
            self.assertIn('cached_rules: 1', format_permission_summary(session_store))
            todo_text = format_todo_summary(session_store)
            self.assertIn('inspect repo', todo_text)
            self.assertIn('completed', todo_text)
            self.assertIn('permissions:', format_session_panel(session_store))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_execute_resume_slash_command(self) -> None:
        tmp_dir = Path('D:/workspace/Anvil/.tmp') / f'session-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            session_store = SessionStore.create(
                root_dir=tmp_dir,
                workspace_root=tmp_dir,
                goal='inspect runtime',
                memory_run_dir=tmp_dir / 'memory',
            )
            session_store.append_event('chat_user', {'role': 'user', 'content': 'hello'})
            session_store.state.permission_stats = {'allow': 1, 'deny': 0, 'ask': 0}
            result = execute_slash_command(
                parse_slash_command('/resume'),
                session_store=session_store,
                tool_specs=builtin_tool_specs(),
            )
            self.assertIn('session_id:', result.output)
            self.assertIn('inspect runtime', result.output)
            self.assertIn('user: hello', result.output)
            self.assertIn('permissions:', result.output)
            self.assertIn('recent_events:', result.output)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_support_history_events_and_tool_filters(self) -> None:
        tmp_dir = Path('D:/workspace/Anvil/.tmp') / f'session-cmds-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            session_store = SessionStore.create(
                root_dir=tmp_dir,
                workspace_root=tmp_dir,
                goal='inspect runtime',
                memory_run_dir=tmp_dir / 'memory',
            )
            session_store.state.last_summary = 'repo inspected'
            session_store.append_event('chat_user', {'role': 'user', 'content': 'one'})
            session_store.append_event('chat_assistant', {'role': 'assistant', 'content': 'two'})
            history_result = execute_slash_command(
                parse_slash_command('/history 1'),
                session_store=session_store,
                tool_specs=builtin_tool_specs(),
            )
            events_result = execute_slash_command(
                parse_slash_command('/events 2'),
                session_store=session_store,
                tool_specs=builtin_tool_specs(),
            )
            tools_result = execute_slash_command(
                parse_slash_command('/tools git'),
                session_store=session_store,
                tool_specs=builtin_tool_specs(),
            )
            summary_result = execute_slash_command(
                parse_slash_command('/summary'),
                session_store=session_store,
                tool_specs=builtin_tool_specs(),
            )
            panel_result = execute_slash_command(
                parse_slash_command('/panel'),
                session_store=session_store,
                tool_specs=builtin_tool_specs(),
            )
            self.assertIn('assistant: two', history_result.output)
            self.assertNotIn('user: one', history_result.output)
            self.assertIn('chat_assistant', events_result.output)
            # Git tools removed from default dispatch (use run_command instead)
            self.assertNotIn('git_status', tools_result.output)
            self.assertNotIn('read_file', tools_result.output)
            self.assertIn('repo inspected', summary_result.output)
            self.assertIn('recent_events:', panel_result.output)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_render_pretty_event_row(self) -> None:
        text = render_event_row(
            {
                'ts': '2026-01-01T00:00:00Z',
                'event': 'step_succeeded',
                'tool_name': 'read_file',
                'permission_decision': 'allow',
                'session_id': 'sess-1',
            }
        )
        self.assertIn('step_succeeded', text)
        self.assertIn('[read_file]', text)
        self.assertIn('permission=allow', text)
        self.assertIn('session=sess-1', text)


if __name__ == '__main__':
    unittest.main()
