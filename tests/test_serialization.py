from __future__ import annotations

import json
import unittest

import _bootstrap  # noqa: F401

from loop_agent.core.serialization import run_result_to_dict, run_result_to_json
from loop_agent.core.types import RunResult, StopReason


class SerializationTests(unittest.TestCase):
    def test_should_serialize_to_dict_without_history(self) -> None:
        result = RunResult(
            final_output='ok',
            state={'v': 1},
            done=True,
            steps=2,
            elapsed_s=0.2,
            history=('a', 'b'),
            stop_reason=StopReason.done,
            error=None,
        )
        payload = run_result_to_dict(result, include_history=False)
        self.assertEqual(payload['stop_reason'], 'done')
        self.assertNotIn('history', payload)

    def test_should_serialize_to_json(self) -> None:
        result = RunResult(
            final_output='ok',
            state={'v': 1},
            done=True,
            steps=2,
            elapsed_s=0.2,
            history=('a', 'b'),
            stop_reason=StopReason.done,
            error=None,
        )
        raw = run_result_to_json(result, include_history=True)
        payload = json.loads(raw)
        self.assertEqual(payload['stop_reason'], 'done')
        self.assertEqual(payload['history'], ['a', 'b'])

    def test_should_extract_compression_state_into_top_level_payload(self) -> None:
        result = RunResult(
            final_output='ok',
            state={
                'compact_summary': 'checkpoint-1',
                'compaction_count': 2,
                'archived_transcripts': ['.transcripts/compact_0002.json'],
                'last_compaction_reason': 'auto',
            },
            done=True,
            steps=2,
            elapsed_s=0.2,
            history=('a', 'b'),
            stop_reason=StopReason.done,
            error=None,
        )

        payload = run_result_to_dict(result, include_history=False)

        self.assertEqual(payload['compression_state']['summary'], 'checkpoint-1')
        self.assertEqual(payload['compression_state']['compaction_count'], 2)
        self.assertEqual(payload['compression_state']['last_compaction_reason'], 'auto')


if __name__ == '__main__':
    unittest.main()
