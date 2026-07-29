from __future__ import annotations

import json
import os
import tempfile
import unittest

import _bootstrap  # noqa: F401

from anvil.errors import validate_goal
from anvil.utils import build_jsonl_observer, resolve_goal


class UtilsErrorsTests(unittest.TestCase):
    def test_should_reject_empty_goal(self) -> None:
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

    def test_should_read_goal_from_utf8_file(self) -> None:
        with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8', suffix='.txt') as file:
            file.write('目标')
            path = file.name
        try:
            self.assertEqual(resolve_goal(goal_file=path), '目标')
        finally:
            os.remove(path)


if __name__ == '__main__':
    unittest.main()
