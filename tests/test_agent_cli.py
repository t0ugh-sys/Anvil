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
from anvil.services.catalog_service import render_skills, render_tools
from anvil.services.replay_service import render_replay, resolve_events_file
from anvil.services.team_service import parse_team_message, parse_teammate
from anvil.session import SessionStore
from anvil.skills import SkillLoader
from anvil.tools import builtin_tool_specs


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
            self.assertIn('[completed] inspect repo', format_todo_summary(session_store))
            self.assertIn('permissions:', format_session_panel(session_store))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

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
            self.assertIn('git_status', tools_result.output)
            self.assertNotIn('read_file', tools_result.output)
            self.assertIn('repo inspected', summary_result.output)
            self.assertIn('recent_events:', panel_result.output)
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

    def test_should_render_catalog_outputs(self) -> None:
        tools_text = render_tools(verbose=False)
        skills_text = render_skills()
        self.assertIn('read_file', tools_text)
        self.assertIn('Available skills:', skills_text)

    def test_should_resolve_and_render_replay(self) -> None:
        tmp_dir = Path('D:/workspace/Anvil/.tmp') / f'replay-svc-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            events_dir = tmp_dir / 'sessions' / 'sess-1'
            events_dir.mkdir(parents=True, exist_ok=True)
            events_file = events_dir / 'events.jsonl'
            events_file.write_text(json.dumps({'ts': '2026-01-01T00:00:00Z', 'event': 'run_started'}) + '\n', encoding='utf-8')
            resolved = resolve_events_file(events_file='', session_id='sess-1', sessions_dir=str(tmp_dir / 'sessions'))
            self.assertEqual(resolved, events_file)
            pretty = render_replay(events_file=events_file, pretty=True, limit=5)
            raw = render_replay(events_file=events_file, pretty=False, limit=None)
            self.assertIn('run_started', pretty)
            self.assertIn('"event": "run_started"', raw)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_parse_team_inputs(self) -> None:
        self.assertEqual(parse_teammate('dev:coder'), ('dev', 'coder'))
        self.assertEqual(parse_team_message('lead=ship it'), ('lead', 'ship it'))
        with self.assertRaises(ValueError):
            parse_teammate('broken')
        with self.assertRaises(ValueError):
            parse_team_message('broken')

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

    def test_should_pretty_replay_events_from_session_file(self) -> None:
        tmp_dir = Path('D:/workspace/Anvil/.tmp') / f'replay-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        events_file = tmp_dir / 'events.jsonl'
        try:
            events_file.write_text(
                '\n'.join(
                    [
                        json.dumps({'ts': '2026-01-01T00:00:00Z', 'event': 'run_started', 'session_id': 'sess-1'}),
                        json.dumps(
                            {
                                'ts': '2026-01-01T00:00:01Z',
                                'event': 'step_succeeded',
                                'tool_name': 'read_file',
                                'permission_decision': 'allow',
                                'session_id': 'sess-1',
                            }
                        ),
                    ]
                ),
                encoding='utf-8',
            )
            parser = build_parser()
            replay_args = parser.parse_args(['replay', '--events-file', str(events_file), '--pretty', '--limit', '1'])
            replay_buffer = io.StringIO()
            with patch('sys.stdout', replay_buffer):
                self.assertEqual(replay_args.handler(replay_args), 0)
            replay_text = replay_buffer.getvalue()
            self.assertIn('step_succeeded', replay_text)
            self.assertIn('[read_file]', replay_text)
            self.assertNotIn('run_started', replay_text)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_parse_team_run_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                'team',
                'run',
                '--workspace',
                '.',
                '--teammate',
                'alice:coder',
                '--message',
                'alice=fix bug',
                '--provider',
                'mock',
                '--model',
                'mock-v3',
                '--output',
                'json',
            ]
        )
        self.assertEqual(args.command, 'team')
        self.assertEqual(args.team_command, 'run')
        self.assertEqual(args.teammate, ['alice:coder'])
        self.assertEqual(args.message, ['alice=fix bug'])

    def test_should_parse_team_send_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ['team', 'send', '--workspace', '.', '--to', 'alice', '--message', 'fix bug']
        )
        self.assertEqual(args.command, 'team')
        self.assertEqual(args.team_command, 'send')
        self.assertEqual(args.to, 'alice')

    def test_should_parse_team_serve_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                'team',
                'serve',
                '--workspace',
                '.',
                '--teammate',
                'alice:coder',
                '--service-timeout-s',
                '10',
            ]
        )
        self.assertEqual(args.command, 'team')
        self.assertEqual(args.team_command, 'serve')
        self.assertEqual(args.service_timeout_s, 10.0)

    def test_should_parse_team_add_task_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                'team',
                'add-task',
                '--workspace',
                '.',
                '--goal',
                'inspect README',
                '--depends-on',
                'task_1',
            ]
        )
        self.assertEqual(args.command, 'team')
        self.assertEqual(args.team_command, 'add-task')
        self.assertEqual(args.depends_on, ['task_1'])

    def test_should_run_team_runtime_with_mock_provider(self) -> None:
        parser = build_parser()
        tmp_dir = Path('tests/.tmp') / f'team-cli-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / 'README.md').write_text('hello', encoding='utf-8')
        try:
            args = parser.parse_args(
                [
                    'team',
                    'run',
                    '--workspace',
                    str(tmp_dir),
                    '--teammate',
                    'alice:coder',
                    '--message',
                    'alice=inspect README',
                    '--provider',
                    'mock',
                    '--model',
                    'mock-v3',
                    '--output',
                    'json',
                    '--service-timeout-s',
                    '2',
                ]
            )
            with patch('sys.stdout'):
                exit_code = args.handler(args)
            self.assertEqual(exit_code, 0)
            self.assertTrue((tmp_dir / '.team' / 'config.json').exists())
            self.assertTrue((tmp_dir / '.team' / 'inbox' / 'alice.jsonl').exists())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_run_team_tasks_with_mock_provider(self) -> None:
        parser = build_parser()
        tmp_dir = Path('tests/.tmp') / f'team-cli-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / 'README.md').write_text('hello', encoding='utf-8')
        try:
            args = parser.parse_args(
                [
                    'team',
                    'run',
                    '--workspace',
                    str(tmp_dir),
                    '--teammate',
                    'alice:coder',
                    '--task',
                    'inspect README',
                    '--task',
                    'patch README',
                    '--provider',
                    'mock',
                    '--model',
                    'mock-v3',
                    '--output',
                    'json',
                    '--service-timeout-s',
                    '2',
                ]
            )
            with patch('sys.stdout') as stdout:
                exit_code = args.handler(args)
            self.assertEqual(exit_code, 0)
            tasks_dir = tmp_dir / '.team' / 'tasks'
            self.assertTrue((tasks_dir / 'task_task_1.json').exists())
            self.assertTrue((tasks_dir / 'task_task_2.json').exists())
            first_task = json.loads((tasks_dir / 'task_task_1.json').read_text(encoding='utf-8'))
            second_task = json.loads((tasks_dir / 'task_task_2.json').read_text(encoding='utf-8'))
            self.assertEqual(first_task['status'], 'completed')
            self.assertEqual(second_task['status'], 'completed')
            output_text = ''.join(call.args[0] for call in stdout.write.call_args_list if call.args)
            payload = json.loads(output_text)
            self.assertEqual(len(payload['tasks']), 2)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_serve_team_and_process_external_message(self) -> None:
        parser = build_parser()
        tmp_dir = Path('tests/.tmp') / f'team-service-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / 'README.md').write_text('hello', encoding='utf-8')
        service_args = parser.parse_args(
            [
                'team',
                'serve',
                '--workspace',
                str(tmp_dir),
                '--teammate',
                'alice:coder',
                '--provider',
                'mock',
                '--model',
                'mock-v3',
                '--output',
                'json',
                '--service-timeout-s',
                '2',
            ]
        )
        send_args = parser.parse_args(
            [
                'team',
                'send',
                '--workspace',
                str(tmp_dir),
                '--to',
                'alice',
                '--message',
                'inspect README',
            ]
        )
        outputs: list[str] = []

        def run_service() -> None:
            with patch('builtins.print', side_effect=lambda *parts, **_: outputs.append(' '.join(str(part) for part in parts))):
                service_args.handler(service_args)

        worker = threading.Thread(target=run_service, daemon=True)
        worker.start()
        try:
            time.sleep(0.2)
            send_args.handler(send_args)
            worker.join(timeout=5.0)
            self.assertFalse(worker.is_alive())
            self.assertTrue(outputs)
            payload = json.loads(outputs[-1])
            self.assertEqual(payload['mode'], 'service')
            self.assertTrue(any(reply['sender'] == 'alice' for reply in payload['replies']))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_serve_team_and_pick_up_added_task(self) -> None:
        parser = build_parser()
        tmp_dir = Path('tests/.tmp') / f'team-service-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / 'README.md').write_text('hello', encoding='utf-8')
        service_args = parser.parse_args(
            [
                'team',
                'serve',
                '--workspace',
                str(tmp_dir),
                '--teammate',
                'alice:coder',
                '--provider',
                'mock',
                '--model',
                'mock-v3',
                '--output',
                'json',
                '--service-timeout-s',
                '2',
            ]
        )
        add_task_args = parser.parse_args(
            [
                'team',
                'add-task',
                '--workspace',
                str(tmp_dir),
                '--goal',
                'inspect README',
                '--output',
                'json',
            ]
        )
        outputs: list[str] = []

        def run_service() -> None:
            with patch('builtins.print', side_effect=lambda *parts, **_: outputs.append(' '.join(str(part) for part in parts))):
                service_args.handler(service_args)

        worker = threading.Thread(target=run_service, daemon=True)
        worker.start()
        try:
            time.sleep(0.2)
            with patch('builtins.print'):
                add_task_args.handler(add_task_args)
            worker.join(timeout=5.0)
            self.assertFalse(worker.is_alive())
            payload = json.loads(outputs[-1])
            self.assertTrue(payload['tasks'])
            self.assertEqual(payload['tasks'][0]['status'], 'completed')
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
