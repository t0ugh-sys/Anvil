from __future__ import annotations

import shutil
import unittest
import uuid
import json
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
            exit_code = _run_code_command(args)
            self.assertEqual(exit_code, 0)
            self.assertTrue((memory_dir / 'r1' / 'events.jsonl').exists())
            self.assertTrue((memory_dir / 'r1' / 'summary.json').exists())
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
            _run_code_command(args)
            self.assertTrue((tmp_dir / '.tasks').exists())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

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


if __name__ == '__main__':
    unittest.main()
