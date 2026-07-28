"""Integration: compression pipeline triggers correctly on long message lists."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import _bootstrap  # noqa: F401

import unittest

from anvil.compression import (
    micro_compact_messages,
    partial_compact_messages,
    estimate_messages_tokens,
    CompactConfig,
)


def _make_messages(n_rounds: int) -> list[dict]:
    msgs = [{'role': 'system', 'content': 'You are a coding assistant.'}]
    for i in range(n_rounds):
        msgs.append({'role': 'user', 'content': f'Task {i}: analyze file_{i}.py'})
        # micro_compact_messages looks for tool_result blocks inside assistant content lists
        msgs.append({'role': 'assistant', 'content': [
            {'type': 'tool_use', 'id': f't{i}', 'name': 'read_file', 'input': {}},
            {'type': 'tool_result', 'tool_use_id': f't{i}', 'content': 'x' * 800},
        ]})
    return msgs


def _extract_tool_results(messages: list[dict]) -> list[dict]:
    return [
        block
        for msg in messages if msg.get('role') == 'assistant'
        for block in (msg.get('content') or [])
        if isinstance(block, dict) and block.get('type') == 'tool_result'
    ]


class TestCompressionE2E(unittest.TestCase):
    def test_micro_compact_truncates_early_tool_results(self):
        msgs = _make_messages(10)
        compressed = micro_compact_messages(msgs, keep_last_results=3, max_result_chars=100)
        results = _extract_tool_results(compressed)
        truncated = [b for b in results if '[Earlier' in str(b.get('content', ''))]
        self.assertGreater(len(truncated), 0, 'Expected early tool results to be truncated')

    def test_micro_compact_preserves_recent_results(self):
        msgs = _make_messages(5)
        compressed = micro_compact_messages(msgs, keep_last_results=3, max_result_chars=200)
        results = _extract_tool_results(compressed)
        self.assertGreaterEqual(len(results), 3)
        # Last 3 must keep original 800-char content
        for block in results[-3:]:
            self.assertEqual(block.get('content'), 'x' * 800)
        # Earlier ones must be truncated
        for block in results[:-3]:
            self.assertIn('[Earlier', str(block.get('content', '')))

    def test_partial_compact_reduces_message_count(self):
        msgs = _make_messages(15)
        tokens_before = estimate_messages_tokens(msgs)
        compressed = partial_compact_messages(msgs, keep_recent_rounds=3)
        tokens_after = estimate_messages_tokens(compressed)
        self.assertLessEqual(tokens_after, tokens_before)

    def test_empty_messages_unchanged(self):
        self.assertEqual(micro_compact_messages([]), [])
        self.assertEqual(partial_compact_messages([]), [])

    def test_compact_config_validates(self):
        config = CompactConfig(max_context_tokens=10000, micro_max_result_chars=200)
        config.validate()
        with self.assertRaises(ValueError):
            CompactConfig(max_context_tokens=0).validate()


if __name__ == '__main__':
    unittest.main()
