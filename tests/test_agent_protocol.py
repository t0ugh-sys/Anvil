from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from anvil.agent_protocol import parse_agent_step


class AgentProtocolTests(unittest.TestCase):
    def test_should_parse_first_json_object_when_model_adds_trailing_text(self) -> None:
        raw = (
            '{"thought":"create file","plan":["write"],'
            '"tool_calls":[{"id":"call_1","name":"write_file",'
            '"arguments":{"path":"D:\\\\workspace\\\\abc\\\\data.json","content":""}}],'
            '"final":null}'
            '\nThe user wants me to create a directory and a file.'
        )

        parsed = parse_agent_step(raw)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.thought, 'create file')
        self.assertEqual(parsed.tool_calls[0].name, 'write_file')
        self.assertEqual(parsed.tool_calls[0].arguments['content'], '')

    def test_should_parse_first_json_object_when_model_adds_leading_text(self) -> None:
        raw = (
            'I will call the tool now.\n'
            '{"thought":"done","plan":[],"tool_calls":[],"final":"ok"}'
        )

        parsed = parse_agent_step(raw)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.final, 'ok')


if __name__ == '__main__':
    unittest.main()
