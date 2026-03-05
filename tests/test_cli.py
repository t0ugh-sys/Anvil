from __future__ import annotations

import argparse
import os
import tempfile
import unittest

import _bootstrap  # noqa: F401

from loop_agent.cli import build_parser, resolve_goal
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

    def test_should_reject_empty_goal(self) -> None:
        args = argparse.Namespace(goal='   ', goal_file=None)
        with self.assertRaises(ValueError):
            resolve_goal(args)


if __name__ == '__main__':
    unittest.main()

