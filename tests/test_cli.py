from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
import uuid
import shutil

import _bootstrap  # noqa: F401

from anvil.cli import build_jsonl_observer, build_parser, execute, resolve_goal
from anvil.steps.registry import build_default_registry


class CliTests(unittest.TestCase):
    def test_should_parse_strategy(self) -> None:
        parser = build_parser(build_default_registry())
        args = parser.parse_args(['--goal', 'x', '--strategy', 'json_stub'])
        self.assertEqual(args.strategy, 'json_stub')

    def test_should_parse_provider_and_model(self) -> None:
        parser = build_parser(build_default_registry())
        args = parser.parse_args(
            [
                '--goal',
                'x',
                '--strategy',
                'json_llm',
                '--provider',
                'mock',
                '--model',
                'qwen-max',
                '--wire-api',
                'responses',
                '--provider-debug',
                '--provider-header',
                'x-foo:bar',
                '--provider-header',
                'x-baz:qux',
                '--fallback-model',
                'gpt-5-codex',
                '--max-retries',
                '3',
                '--retry-backoff-s',
                '1.5',
                '--retry-http-code',
                '502',
            ]
        )
        self.assertEqual(args.strategy, 'json_llm')
        self.assertEqual(args.provider, 'mock')
        self.assertEqual(args.model, 'qwen-max')
        self.assertEqual(args.wire_api, 'responses')
        self.assertTrue(args.provider_debug)
        self.assertEqual(args.provider_header, ['x-foo:bar', 'x-baz:qux'])
        self.assertEqual(args.fallback_model, ['gpt-5-codex'])
        self.assertEqual(args.max_retries, 3)
        self.assertEqual(args.retry_backoff_s, 1.5)
        self.assertEqual(args.retry_http_code, [502])

    def test_should_read_goal_from_utf8_file(self) -> None:
        parser = build_parser(build_default_registry())
        with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8', suffix='.txt') as file:
            file.write('目标')
            path = file.name
        try:
            args = parser.parse_args(['--goal-file', path])
            self.assertEqual(resolve_goal(goal_file=args.goal_file), '目标')
        finally:
            os.remove(path)

    def test_should_parse_json_output_options(self) -> None:
        parser = build_parser(build_default_registry())
        args = parser.parse_args(['--goal', 'x', '--output', 'json', '--include-history'])
        self.assertEqual(args.output, 'json')
        self.assertTrue(args.include_history)

    def test_should_parse_observer_and_exit_options(self) -> None:
        parser = build_parser(build_default_registry())
        args = parser.parse_args(
            [
                '--goal',
                'x',
                '--observer-file',
                'events.jsonl',
                '--exit-on-failure',
                '--memory-dir',
                'memory',
                '--run-id',
                'r1',
            ]
        )
        self.assertEqual(args.observer_file, 'events.jsonl')
        self.assertTrue(args.exit_on_failure)
        self.assertTrue(args.record_run)
        self.assertEqual(args.memory_dir, 'memory')
        self.assertEqual(args.run_id, 'r1')

    def test_should_reject_empty_goal(self) -> None:
        from anvil.errors import validate_goal
        with self.assertRaises(Exception):
            validate_goal('   ')

    def test_should_write_observer_jsonl(self) -> None:
        with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8', suffix='.jsonl') as file:
            path = file.name
        try:
            observer = build_jsonl_observer(path)
            observer('step_started', {'step': 1})
            with open(path, 'r', encoding='utf-8') as file:
                line = file.readline().strip()
            payload = json.loads(line)
            self.assertEqual(payload['event'], 'step_started')
            self.assertEqual(payload['payload']['step'], 1)
        finally:
            os.remove(path)

    def test_should_return_nonzero_when_exit_on_failure_and_not_done(self) -> None:
        parser = build_parser(build_default_registry())
        tmp_dir = Path('tests/.tmp') / f'run-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        memory_dir = tmp_dir / 'memory'
        args = parser.parse_args(
            [
                '--goal',
                'x',
                '--strategy',
                'json_stub',
                '--max-steps',
                '1',
                '--exit-on-failure',
                '--output',
                'json',
                '--memory-dir',
                str(memory_dir),
                '--run-id',
                'r1',
            ]
        )
        try:
            rendered, exit_code = execute(args, build_default_registry())
            self.assertEqual(exit_code, 1)
            payload = json.loads(rendered)
            self.assertEqual(payload['stop_reason'], 'max_steps')
            self.assertIn('memory_state', payload)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_return_zero_when_not_done_but_exit_not_required(self) -> None:
        parser = build_parser(build_default_registry())
        tmp_dir = Path('tests/.tmp') / f'run-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        memory_dir = tmp_dir / 'memory'
        args = parser.parse_args(
            ['--goal', 'x', '--strategy', 'json_stub', '--max-steps', '1', '--memory-dir', str(memory_dir)]
        )
        try:
            _, exit_code = execute(args, build_default_registry())
            self.assertEqual(exit_code, 0)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_execute_json_llm_strategy_with_mock_provider(self) -> None:
        parser = build_parser(build_default_registry())
        tmp_dir = Path('tests/.tmp') / f'run-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        memory_dir = tmp_dir / 'memory'
        try:
            args = parser.parse_args(
                [
                    '--goal',
                    'x',
                    '--strategy',
                    'json_llm',
                    '--provider',
                    'mock',
                    '--model',
                    'mock-v2',
                    '--max-steps',
                    '5',
                    '--output',
                    'json',
                    '--memory-dir',
                    str(memory_dir),
                ]
            )
            rendered, exit_code = execute(args, build_default_registry())
            self.assertEqual(exit_code, 0)
            payload = json.loads(rendered)
            self.assertEqual(payload['stop_reason'], 'done')
            self.assertEqual(payload['done'], True)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_create_run_directory_by_default(self) -> None:
        parser = build_parser(build_default_registry())
        tmp_dir = Path('tests/.tmp') / f'run-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        memory_dir = tmp_dir / 'memory'
        try:
            args = parser.parse_args(
                [
                    '--goal',
                    'x',
                    '--runs-dir',
                    str(tmp_dir),
                    '--output',
                    'json',
                    '--memory-dir',
                    str(memory_dir),
                    '--run-id',
                    'r2',
                ]
            )
            rendered, exit_code = execute(args, build_default_registry())
            self.assertEqual(exit_code, 0)
            payload = json.loads(rendered)
            run_dir = payload.get('run_dir', '')
            self.assertTrue(run_dir)
            self.assertTrue(Path(run_dir).exists())
            self.assertTrue((Path(run_dir) / 'events.jsonl').exists())
            self.assertTrue((Path(run_dir) / 'summary.json').exists())
            self.assertTrue((memory_dir / 'r2' / 'events.jsonl').exists())
            self.assertTrue((memory_dir / 'r2' / 'state.json').exists())
            self.assertTrue((memory_dir / 'r2' / 'summary.json').exists())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_not_create_run_directory_when_disabled(self) -> None:
        parser = build_parser(build_default_registry())
        tmp_dir = Path('tests/.tmp') / f'run-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        memory_dir = tmp_dir / 'memory'
        try:
            args = parser.parse_args(
                [
                    '--goal',
                    'x',
                    '--runs-dir',
                    str(tmp_dir),
                    '--no-record-run',
                    '--output',
                    'json',
                    '--memory-dir',
                    str(memory_dir),
                    '--run-id',
                    'r3',
                ]
            )
            rendered, exit_code = execute(args, build_default_registry())
            self.assertEqual(exit_code, 0)
            payload = json.loads(rendered)
            self.assertNotIn('run_dir', payload)
            self.assertTrue((memory_dir / 'r3' / 'events.jsonl').exists())
            self.assertTrue((memory_dir / 'r3' / 'state.json').exists())
            self.assertTrue((memory_dir / 'r3' / 'summary.json').exists())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
