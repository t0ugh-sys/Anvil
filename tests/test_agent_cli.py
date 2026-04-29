from __future__ import annotations

import shutil
import unittest
import uuid
import json
import io
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401

from anvil.agent_cli import _build_coding_decider, _run_code_command, _should_launch_interactive, build_parser
from anvil.commands import execute_slash_command, parse_slash_command
from anvil.commands.slash import render_runtime_panel, render_session_header, render_session_status
from anvil.session import SessionState, SessionStore
from anvil.services.cli_commands import _filter_replay_rows, _render_pretty_replay
from anvil.services.chat_runtime import InteractiveRuntime, TurnExecution
from anvil.services.runtime_config import RuntimeConfigManager
from anvil.skills import SkillLoader
from anvil.tools import builtin_tool_specs
from anvil.tool_views import render_tool_overview


class AgentCliTests(unittest.TestCase):
    def test_should_only_include_skill_metadata_in_prompt(self) -> None:
        loader = SkillLoader()
        self.assertTrue(loader.load('files'))
        captured = {}

        def fake_invoke(prompt: str) -> str:
            captured['prompt'] = prompt
            return '{"thought":"done","plan":[],"tool_calls":[],"final":"done"}'

        parser = build_parser()
        args = parser.parse_args(['code', '--goal', 'x', '--provider', 'mock', '--model', 'mock-v3'])

        with patch('anvil.agent_cli.build_invoke_from_args', return_value=fake_invoke):
            decider = _build_coding_decider(args, loader)
            decider('goal', tuple(), tuple(), {}, tuple())

        prompt = captured['prompt']
        self.assertIn('Available skills:', prompt)
        self.assertIn('- files: Read, write, patch, and search files', prompt)
        self.assertNotIn('# Anvil Skills', prompt)

    def test_should_describe_tool_use_loop_in_root_help(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()
        self.assertIn('tool-use feedback loop', help_text)

    def test_should_launch_interactive_without_subcommand(self) -> None:
        self.assertTrue(_should_launch_interactive([]))
        self.assertTrue(_should_launch_interactive(['--session-id', 's1']))
        self.assertFalse(_should_launch_interactive(['tools']))
        self.assertFalse(_should_launch_interactive(['code', '--goal', 'x']))

    def test_should_list_tools_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(['tools'])
        self.assertEqual(args.command, 'tools')

    def test_should_parse_help_and_resume_slash_commands(self) -> None:
        self.assertEqual(parse_slash_command('/help').name, 'help')
        self.assertEqual(parse_slash_command('/resume now').argument, 'now')
        self.assertEqual(parse_slash_command('/status').name, 'status')
        self.assertEqual(parse_slash_command('/provider anthropic').argument, 'anthropic')
        self.assertEqual(parse_slash_command('/config').name, 'config')
        self.assertEqual(parse_slash_command('/wire-api responses').argument, 'responses')
        self.assertIsNone(parse_slash_command('plain text'))

    def test_should_describe_tool_use_loop_in_code_help(self) -> None:
        parser = build_parser()
        code_parser = build_parser()._subparsers._group_actions[0].choices['code']
        self.assertIn('tool-use feedback loop', code_parser.format_help())

    def test_should_show_examples_and_groups_in_code_help(self) -> None:
        code_parser = build_parser()._subparsers._group_actions[0].choices['code']
        help_text = code_parser.format_help()
        self.assertIn('Examples:', help_text)
        self.assertIn('execution:', help_text)
        self.assertIn('provider:', help_text)
        self.assertIn('memory and artifacts:', help_text)
        self.assertIn('tool dispatch:', help_text)

    def test_should_parse_doctor_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                'doctor',
                '--base-url',
                'https://example.com/v1',
                '--model',
                'gpt-5.3-codex',
                '--wire-api',
                'responses',
            ]
        )
        self.assertEqual(args.command, 'doctor')
        self.assertEqual(args.base_url, 'https://example.com/v1')

    def test_should_parse_replay_filter_option(self) -> None:
        parser = build_parser()
        args = parser.parse_args(['replay', '--pretty', '--filter', 'permissions-only'])
        self.assertEqual(args.command, 'replay')
        self.assertEqual(args.filter, 'permissions-only')

    def test_should_render_interactive_turn_footer(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'chat-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            session_store = SessionStore.create(
                root_dir=tmp_dir,
                workspace_root=tmp_dir,
                goal='inspect runtime',
                memory_run_dir=tmp_dir / 'memory',
            )
            runtime = InteractiveRuntime(
                session_store=session_store,
                tool_specs=builtin_tool_specs(),
                run_turn=lambda _: TurnExecution(
                    output='done',
                    stop_reason='max_steps',
                    steps=2,
                    permission_delta={'allow': 1, 'ask': 1, 'deny': 0},
                    blocked_tool_name='run_command',
                    blocked_tool_reason='approval required',
                ),
                runtime_config_manager=None,
                stdin=io.StringIO('hello\n'),
                stdout=io.StringIO(),
            )
            self.assertEqual(runtime.run(), 0)
            rendered = runtime.stdout.getvalue()
            self.assertIn('done', rendered)
            self.assertIn('[turn] stop_reason=max_steps steps=2 permissions=allow+1,ask+1', rendered)
            self.assertIn('blocked=run_command (approval required)', rendered)
            self.assertIn('session=turns:1 messages:2 commands:0 steps:0', rendered)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_run_team_add_task_command(self) -> None:
        parser = build_parser()
        tmp_dir = Path('tests/.tmp') / f'team-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            args = parser.parse_args(
                [
                    'team',
                    'add-task',
                    '--workspace',
                    str(tmp_dir),
                    '--team-dir',
                    '.team-dev',
                    '--goal',
                    'inspect runtime',
                    '--title',
                    'Inspect Runtime',
                    '--output',
                    'json',
                ]
            )
            buffer = io.StringIO()
            with patch('sys.stdout', buffer):
                self.assertEqual(args.handler(args), 0)
            payload = json.loads(buffer.getvalue())
            self.assertIn('.team-dev', payload['team_dir'])
            self.assertEqual(payload['task']['goal'], 'inspect runtime')
            self.assertEqual(payload['task']['title'], 'Inspect Runtime')
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_execute_resume_slash_command(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'session-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            session_store = SessionStore.create(
                root_dir=tmp_dir,
                workspace_root=tmp_dir,
                goal='inspect runtime',
                memory_run_dir=tmp_dir / 'memory',
            )
            class Args:
                provider = 'mock'
                model = 'mock-model'
                base_url = ''
                wire_api = 'chat_completions'
            runtime_config_manager = RuntimeConfigManager.from_args(session_store=session_store, args=Args())
            session_store.append_event('chat_user', {'role': 'user', 'content': 'hello'})
            session_store.append_event('chat_command', {'command': 'status', 'argument': ''})
            result = execute_slash_command(
                parse_slash_command('/resume'),
                session_store=session_store,
                tool_specs=builtin_tool_specs(),
                runtime_config_manager=runtime_config_manager,
            )
            self.assertIn('session_id:', result.output)
            self.assertIn('inspect runtime', result.output)
            self.assertIn('user: hello', result.output)
            self.assertIn('provider: mock', result.output)
            self.assertIn('turn_count: 1', result.output)
            self.assertIn('message_count: 1', result.output)
            self.assertIn('command_count: 1', result.output)
            self.assertIn('last_user_message: hello', result.output)
            config_result = execute_slash_command(
                parse_slash_command('/config'),
                session_store=session_store,
                tool_specs=builtin_tool_specs(),
                runtime_config_manager=runtime_config_manager,
            )
            self.assertIn('Anvil runtime session', config_result.output)
            self.assertIn('runtime_config:', config_result.output)
            self.assertIn('permission_mode_cache_entries:', config_result.output)
            change_result = execute_slash_command(
                parse_slash_command('/provider anthropic'),
                session_store=session_store,
                tool_specs=builtin_tool_specs(),
                runtime_config_manager=runtime_config_manager,
            )
            self.assertIn('provider: anthropic', change_result.output)
            self.assertEqual(session_store.state.runtime_config.get('provider'), 'anthropic')
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_run_code_with_mock_provider(self) -> None:
        parser = build_parser()
        tmp_dir = Path('tests/.tmp') / f'ocli-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        readme = tmp_dir / 'README.md'
        readme.write_text('hello', encoding='utf-8')
        memory_dir = tmp_dir / 'memory'
        try:
            args = parser.parse_args(
                [
                    'code',
                    '--goal',
                    'read workspace then finalize',
                    '--workspace',
                    str(tmp_dir),
                    '--provider',
                    'mock',
                    '--model',
                    'mock-v3',
                    '--memory-dir',
                    str(memory_dir),
                    '--run-id',
                    'r1',
                    '--output',
                    'json',
                ]
            )
            with patch('sys.stdout', io.StringIO()):
                exit_code = _run_code_command(args)
            self.assertEqual(exit_code, 0)
            self.assertTrue((memory_dir / 'r1' / 'events.jsonl').exists())
            self.assertTrue((memory_dir / 'r1' / 'summary.json').exists())
            session_root = tmp_dir / '.anvil' / 'sessions'
            sessions = [item for item in session_root.iterdir() if item.is_dir()]
            self.assertEqual(len(sessions), 1)
            self.assertTrue((sessions[0] / 'session.json').exists())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_parse_code_memory_and_record_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                'code',
                '--goal',
                'x',
                '--memory-dir',
                'mem',
                '--run-id',
                'run-1',
                '--summarize-every',
                '3',
                '--observer-file',
                'events.jsonl',
                '--no-record-run',
                '--include-history',
                '--tasks-dir',
                '.tasks-dev',
                '--transcripts-dir',
                '.transcripts-dev',
                '--max-context-tokens',
                '2048',
                '--micro-compact-keep',
                '4',
                '--recent-transcript-entries',
                '6',
                '--session-id',
                's1',
                '--sessions-dir',
                '.sessions',
                '--permission-mode',
                'unsafe',
            ]
        )
        self.assertEqual(args.memory_dir, 'mem')
        self.assertEqual(args.run_id, 'run-1')
        self.assertEqual(args.summarize_every, 3)
        self.assertEqual(args.observer_file, 'events.jsonl')
        self.assertFalse(args.record_run)
        self.assertTrue(args.include_history)
        self.assertEqual(args.tasks_dir, '.tasks-dev')
        self.assertEqual(args.transcripts_dir, '.transcripts-dev')
        self.assertEqual(args.max_context_tokens, 2048)
        self.assertEqual(args.micro_compact_keep, 4)
        self.assertEqual(args.recent_transcript_entries, 6)
        self.assertEqual(args.session_id, 's1')
        self.assertEqual(args.sessions_dir, '.sessions')
        self.assertEqual(args.permission_mode, 'unsafe')

    def test_should_record_structured_tool_events(self) -> None:
        parser = build_parser()
        tmp_dir = Path('tests/.tmp') / f'ocli-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        readme = tmp_dir / 'README.md'
        readme.write_text('hello', encoding='utf-8')
        observer_file = tmp_dir / 'events.jsonl'
        try:
            args = parser.parse_args(
                [
                    'code',
                    '--goal',
                    'read workspace then finalize',
                    '--workspace',
                    str(tmp_dir),
                    '--provider',
                    'mock',
                    '--model',
                    'mock-v3',
                    '--observer-file',
                    str(observer_file),
                    '--output',
                    'json',
                ]
            )
            with patch('sys.stdout', io.StringIO()):
                _run_code_command(args)
            rows = []
            for line in observer_file.read_text(encoding='utf-8').splitlines():
                rows.append(json.loads(line))
            step_succeeded = [row for row in rows if row.get('event') == 'step_succeeded']
            self.assertTrue(step_succeeded)
            metadata = step_succeeded[0]['payload'].get('metadata', {})
            self.assertIn('tool_calls', metadata)
            self.assertIn('tool_results', metadata)
            self.assertIn('todo_state', metadata)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_create_task_store_directory_by_default(self) -> None:
        parser = build_parser()
        tmp_dir = Path('tests/.tmp') / f'ocli-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / 'README.md').write_text('hello', encoding='utf-8')
        try:
            args = parser.parse_args(
                [
                    'code',
                    '--goal',
                    'read workspace then finalize',
                    '--workspace',
                    str(tmp_dir),
                    '--provider',
                    'mock',
                    '--model',
                    'mock-v3',
                ]
            )
            with patch('sys.stdout', io.StringIO()):
                _run_code_command(args)
            self.assertTrue((tmp_dir / '.tasks').exists())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_resume_existing_session_without_goal(self) -> None:
        parser = build_parser()
        tmp_dir = Path('tests/.tmp') / f'ocli-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / 'README.md').write_text('hello', encoding='utf-8')
        memory_dir = tmp_dir / 'memory'
        sessions_dir = tmp_dir / '.sessions'
        try:
            args = parser.parse_args(
                [
                    'code',
                    '--goal',
                    'read workspace then finalize',
                    '--workspace',
                    str(tmp_dir),
                    '--provider',
                    'mock',
                    '--model',
                    'mock-v3',
                    '--memory-dir',
                    str(memory_dir),
                    '--sessions-dir',
                    str(sessions_dir),
                    '--output',
                    'json',
                ]
            )
            buffer = io.StringIO()
            with patch('sys.stdout', buffer):
                self.assertEqual(_run_code_command(args), 0)
            first_payload = json.loads(buffer.getvalue())
            session_id = first_payload['session_id']

            args2 = parser.parse_args(
                [
                    'code',
                    '--session-id',
                    session_id,
                    '--workspace',
                    str(tmp_dir),
                    '--provider',
                    'mock',
                    '--model',
                    'mock-v3',
                    '--memory-dir',
                    str(memory_dir),
                    '--sessions-dir',
                    str(sessions_dir),
                    '--output',
                    'json',
                ]
            )
            buffer2 = io.StringIO()
            with patch('sys.stdout', buffer2):
                self.assertEqual(_run_code_command(args2), 0)
            second_payload = json.loads(buffer2.getvalue())
            self.assertEqual(second_payload['session_id'], session_id)
            self.assertEqual(second_payload['memory_run_dir'], first_payload['memory_run_dir'])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_replay_events_from_session_id(self) -> None:
        parser = build_parser()
        tmp_dir = Path('tests/.tmp') / f'ocli-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / 'README.md').write_text('hello', encoding='utf-8')
        sessions_dir = tmp_dir / '.sessions'
        try:
            args = parser.parse_args(
                [
                    'code',
                    '--goal',
                    'read workspace then finalize',
                    '--workspace',
                    str(tmp_dir),
                    '--provider',
                    'mock',
                    '--model',
                    'mock-v3',
                    '--sessions-dir',
                    str(sessions_dir),
                    '--output',
                    'json',
                ]
            )
            buffer = io.StringIO()
            with patch('sys.stdout', buffer):
                self.assertEqual(_run_code_command(args), 0)
            payload = json.loads(buffer.getvalue())
            replay_args = parser.parse_args(
                ['replay', '--session-id', payload['session_id'], '--sessions-dir', str(sessions_dir)]
            )
            replay_buffer = io.StringIO()
            with patch('sys.stdout', replay_buffer):
                self.assertEqual(replay_args.handler(replay_args), 0)
            replay_text = replay_buffer.getvalue()
            self.assertIn('"event": "run_started"', replay_text)
            self.assertIn(payload['session_id'], replay_text)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_pretty_render_replay_rows(self) -> None:
        output = _render_pretty_replay(
            [
                {
                    'ts': '2026-04-29T10:00:00Z',
                    'event': 'run_started',
                    'payload': {'goal': 'inspect runtime'},
                    'session_id': 's1',
                },
                {
                    'ts': '2026-04-29T10:00:01Z',
                    'event': 'step_succeeded',
                    'step': 1,
                    'tool_name': 'todo_write',
                    'permission_decision': 'allow',
                    'payload': {'output': 'updated config'},
                    'session_id': 's1',
                },
                {
                    'ts': '2026-04-29T10:00:02Z',
                    'event': 'run_finished',
                    'payload': {'done': True, 'stop_reason': 'finished', 'steps': 1},
                    'session_id': 's1',
                },
            ]
        )
        self.assertIn('run_started', output)
        self.assertIn('inspect runtime', output)
        self.assertIn('step=1 step_succeeded', output)
        self.assertIn('todo_write', output)
        self.assertIn('allow', output)
        self.assertIn('run_finished', output)

    def test_should_filter_replay_rows_by_permissions(self) -> None:
        rows = [
            {'event': 'run_started', 'payload': {'goal': 'inspect runtime'}},
            {
                'event': 'step_succeeded',
                'tool_name': 'run_command',
                'permission_decision': 'ask',
                'payload': {
                    'metadata': {
                        'tool_results': [
                            {'permission_decision': 'ask', 'permission_reason': 'approval required'}
                        ]
                    }
                },
            },
            {'event': 'chat_user', 'payload': {'content': 'hello'}},
        ]
        filtered = _filter_replay_rows(rows, 'permissions-only')
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['tool_name'], 'run_command')

    def test_should_filter_replay_rows_by_errors(self) -> None:
        rows = [
            {'event': 'run_started', 'payload': {'goal': 'inspect runtime'}},
            {
                'event': 'step_succeeded',
                'tool_name': 'run_command',
                'payload': {
                    'metadata': {
                        'tool_results': [
                            {'error': 'exit=1', 'permission_decision': 'allow'}
                        ]
                    }
                },
            },
            {'event': 'run_finished', 'payload': {'done': False, 'stop_reason': 'step_error'}},
        ]
        filtered = _filter_replay_rows(rows, 'errors-only')
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]['event'], 'step_succeeded')
        self.assertEqual(filtered[1]['event'], 'run_finished')

    def test_should_render_session_status_with_todos_and_permissions(self) -> None:
        state = SessionState(
            session_id='s1',
            workspace_root='D:/workspace/Anvil',
            goal='inspect runtime',
            status='active',
            created_at='2026-04-29T10:00:00Z',
            updated_at='2026-04-29T10:01:00Z',
            last_activity_at='2026-04-29T10:01:00Z',
            history_tail=['user: hello', 'assistant: ok'],
            tool_history=[
                {'name': 'read_file', 'ok': True, 'permission_decision': 'allow'},
                {'name': 'run_command_async', 'ok': False, 'permission_decision': 'ask'},
            ],
            todo_state={'lines': ['[x] inspect repo', '[-] switch model']},
            permission_cache={},
            runtime_config={},
            memory_run_dir='.anvil/runs/r1',
            artifacts_dir='.anvil/runs/r1',
            last_summary='working through runtime config',
            turn_count=2,
            message_count=4,
            command_count=1,
            step_count=3,
            last_user_message='hello',
            last_stop_reason='max_steps',
            last_blocked_tool_name='run_command_async',
            last_blocked_tool_reason='approval required',
            permission_stats={'allow': 3, 'deny': 1, 'ask': 2},
        )
        output = render_session_status(
            state,
            runtime_config_text='runtime_config:\nprovider: anthropic\nmodel: claude-3-opus-20240229',
        )
        self.assertIn('session_id: s1', output)
        self.assertIn('turn_count: 2', output)
        self.assertIn('message_count: 4', output)
        self.assertIn('command_count: 1', output)
        self.assertIn('step_count: 3', output)
        self.assertIn('last_activity_at: 2026-04-29T10:01:00Z', output)
        self.assertIn('last_user_message: hello', output)
        self.assertIn('last_stop_reason: max_steps', output)
        self.assertIn('last_blocked_tool: run_command_async', output)
        self.assertIn('last_blocked_reason: approval required', output)
        self.assertIn('permission_stats: allow=3 deny=1 ask=2', output)
        self.assertIn('todo_state:\n[x] inspect repo', output)
        self.assertIn('recent_tools:\n- read_file [ok] permission=allow', output)
        self.assertIn('runtime_config:\nprovider: anthropic', output)

    def test_should_render_runtime_panel(self) -> None:
        state = SessionState(
            session_id='s1',
            workspace_root='D:/workspace/Anvil',
            goal='inspect runtime',
            status='active',
            created_at='2026-04-29T10:00:00Z',
            updated_at='2026-04-29T10:01:00Z',
            last_activity_at='2026-04-29T10:01:00Z',
            permission_cache={'run_command:execute': 'ask'},
            memory_run_dir='.anvil/runs/r1',
            artifacts_dir='.anvil/runs/r1',
            last_stop_reason='max_steps',
            turn_count=3,
            message_count=5,
            command_count=2,
            step_count=4,
        )
        output = render_runtime_panel(
            state,
            runtime_config_text='runtime_config:\nprovider: anthropic\nmodel: claude-3-opus-20240229',
        )
        self.assertIn('Anvil runtime session s1', output)
        self.assertIn('permission_mode_cache_entries: 1', output)
        self.assertIn('last_stop_reason: max_steps', output)
        self.assertIn('runtime_config:\nprovider: anthropic', output)

    def test_should_render_session_header_with_counts(self) -> None:
        state = SessionState(
            session_id='s1',
            workspace_root='D:/workspace/Anvil',
            goal='inspect runtime',
            status='active',
            created_at='2026-04-29T10:00:00Z',
            updated_at='2026-04-29T10:01:00Z',
            turn_count=3,
            message_count=7,
            command_count=2,
            step_count=5,
        )
        output = render_session_header(state, runtime_label='interactive')
        self.assertIn('Anvil interactive session s1', output)
        self.assertIn('workspace: D:/workspace/Anvil', output)
        self.assertIn('turns=3 messages=7 commands=2 steps=5', output)

    def test_should_render_tool_overview_grouped_by_capability(self) -> None:
        output = render_tool_overview(builtin_tool_specs())
        self.assertIn('[read]', output)
        self.assertIn('[write]', output)
        self.assertIn('[execute]', output)
        self.assertIn('read_file risk=low', output)
        self.assertIn('run_command risk=high', output)


if __name__ == '__main__':
    unittest.main()
