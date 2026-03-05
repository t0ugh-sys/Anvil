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

from loop_agent.cli import build_jsonl_observer, build_parser, execute, resolve_goal
from loop_agent.steps.registry import build_default_registry


class CliTests(unittest.TestCase):
    def test_should_parse_strategy(self) -> None:
        parser = build_parser(build_default_registry())
        args = parser.parse_args(['--goal', 'x', '--strategy', 'json_stub'])
        self.assertEqual(args.strategy, 'json_stub')

    def test_should_read_goal_from_utf8_file(self) -> None:
        parser = build_parser(build_default_registry())
        with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8', suffix='.txt') as file:
            file.write('目标')
            path = file.name
        try:
            args = parser.parse_args(['--goal-file', path])
            self.assertEqual(resolve_goal(args), '目标')
        finally:
            os.remove(path)

    def test_should_parse_json_output_options(self) -> None:
        parser = build_parser(build_default_registry())
        args = parser.parse_args(['--goal', 'x', '--output', 'json', '--include-history'])
        self.assertEqual(args.output, 'json')
        self.assertTrue(args.include_history)

    def test_should_parse_observer_and_exit_options(self) -> None:
        parser = build_parser(build_default_registry())
        args = parser.parse_args(['--goal', 'x', '--observer-file', 'events.jsonl', '--exit-on-failure'])
        self.assertEqual(args.observer_file, 'events.jsonl')
        self.assertTrue(args.exit_on_failure)
        self.assertTrue(args.record_run)

    def test_should_reject_empty_goal(self) -> None:
        args = argparse.Namespace(goal='   ', goal_file=None)
        with self.assertRaises(ValueError):
            resolve_goal(args)

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
            ]
        )
        rendered, exit_code = execute(args, build_default_registry())
        self.assertEqual(exit_code, 1)
        payload = json.loads(rendered)
        self.assertEqual(payload['stop_reason'], 'max_steps')

    def test_should_return_zero_when_not_done_but_exit_not_required(self) -> None:
        parser = build_parser(build_default_registry())
        args = parser.parse_args(['--goal', 'x', '--strategy', 'json_stub', '--max-steps', '1'])
        _, exit_code = execute(args, build_default_registry())
        self.assertEqual(exit_code, 0)

    def test_should_create_run_directory_by_default(self) -> None:
        parser = build_parser(build_default_registry())
        tmp_dir = Path('tests/.tmp') / f'run-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            args = parser.parse_args(['--goal', 'x', '--runs-dir', str(tmp_dir), '--output', 'json'])
            rendered, exit_code = execute(args, build_default_registry())
            self.assertEqual(exit_code, 0)
            payload = json.loads(rendered)
            run_dir = payload.get('run_dir', '')
            self.assertTrue(run_dir)
            self.assertTrue(Path(run_dir).exists())
            self.assertTrue((Path(run_dir) / 'events.jsonl').exists())
            self.assertTrue((Path(run_dir) / 'summary.json').exists())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_not_create_run_directory_when_disabled(self) -> None:
        parser = build_parser(build_default_registry())
        tmp_dir = Path('tests/.tmp') / f'run-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            args = parser.parse_args(['--goal', 'x', '--runs-dir', str(tmp_dir), '--no-record-run', '--output', 'json'])
            rendered, exit_code = execute(args, build_default_registry())
            self.assertEqual(exit_code, 0)
            payload = json.loads(rendered)
            self.assertNotIn('run_dir', payload)
            self.assertEqual(list(Path(tmp_dir).iterdir()), [])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
