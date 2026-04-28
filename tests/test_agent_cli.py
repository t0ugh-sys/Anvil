from __future__ import annotations

import shutil
import unittest
import uuid
import json
import io
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401

from loop_agent.agent_cli import _build_coding_decider, _run_code_command, build_parser
from loop_agent.skills import SkillLoader


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

        with patch('loop_agent.agent_cli.build_invoke_from_args', return_value=fake_invoke):
            decider = _build_coding_decider(args, loader)
            decider('goal', tuple(), tuple(), {}, tuple())

        prompt = captured['prompt']
        self.assertIn('Available skills:', prompt)
        self.assertIn('- files: Read, write, patch, and search files', prompt)
        self.assertNotIn('# LoopAgent Skills', prompt)

    def test_should_describe_tool_use_loop_in_root_help(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()
        self.assertIn('tool-use feedback loop', help_text)

    def test_should_list_tools_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(['tools'])
        self.assertEqual(args.command, 'tools')

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
            session_root = tmp_dir / '.loopagent' / 'sessions'
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


if __name__ == '__main__':
    unittest.main()
