from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from loop_agent.protocols.json_decision import parse_json_decision


class JsonProtocolTests(unittest.TestCase):
    def test_should_parse_plain_json(self) -> None:
        decision = parse_json_decision('{"answer":"hi","done":true}')
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.answer, 'hi')
        self.assertTrue(decision.done)

    def test_should_parse_json_code_fence(self) -> None:
        decision = parse_json_decision("```json\n{\"answer\":\"ok\",\"done\":false}\n```")
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.answer, 'ok')
        self.assertFalse(decision.done)

    def test_should_return_none_on_invalid_json(self) -> None:
        self.assertIsNone(parse_json_decision('not json'))


if __name__ == '__main__':
    unittest.main()

