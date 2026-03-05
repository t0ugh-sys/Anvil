from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401

from loop_agent.agent_cli import _run_code_command, build_parser


class AgentCliTests(unittest.TestCase):
    def test_should_list_tools_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(['tools'])
        self.assertEqual(args.command, 'tools')

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
                    '--output',
                    'json',
                ]
            )
            exit_code = _run_code_command(args)
            self.assertEqual(exit_code, 0)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
