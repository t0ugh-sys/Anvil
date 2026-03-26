from __future__ import annotations

import unittest
from pathlib import Path
import shutil
import uuid

import _bootstrap  # noqa: F401

from loop_agent.memory.jsonl_store import JsonlMemoryStore


class MemoryStoreTests(unittest.TestCase):
    def test_should_summarize_and_return_context(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'memory-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            store = JsonlMemoryStore(memory_dir=tmp_dir, summarize_every=2)
            store.on_event('run_started', {'goal': 'g1', 'facts': ['f1']})
            store.on_event('step_started', {'step': 1, 'plan': ['p1', 'p2']})
            store.on_event('step_succeeded', {'output': 'done1'})
            store.on_event('run_finished', {'done': True})

            context = store.load_context(goal='g1', last_k_steps=5)
            self.assertEqual(context.state_summary.get('goal'), 'g1')
            self.assertIn('f1', context.state_summary.get('facts', []))
            self.assertIn('done1', context.state_summary.get('work_done', []))
            self.assertEqual(context.last_steps, ('done1',))
            self.assertTrue((tmp_dir / 'state.json').exists())
            self.assertTrue((tmp_dir / 'summary.json').exists())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_load_recent_steps_from_state_history(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'memory-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            store = JsonlMemoryStore(memory_dir=tmp_dir, summarize_every=10)
            store.on_event('step_succeeded', {'output': 'done1'})
            store.on_event('step_succeeded', {'output': 'done2'})
            (tmp_dir / 'events.jsonl').unlink()

            context = store.load_context(goal='g1', last_k_steps=2)

            self.assertEqual(context.last_steps, ('done1', 'done2'))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_validate_summarize_every(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'memory-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            with self.assertRaises(ValueError):
                JsonlMemoryStore(memory_dir=tmp_dir, summarize_every=0)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_persist_compression_checkpoint_in_summary(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'memory-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            store = JsonlMemoryStore(memory_dir=tmp_dir, summarize_every=1)
            store.on_event('run_started', {'goal': 'g1', 'facts': []})
            store.on_event(
                'step_succeeded',
                {
                    'output': 'done1',
                    'metadata': {
                        'compression_state': {
                            'summary': 'checkpoint-1',
                            'compaction_count': 1,
                            'archived_transcripts': ['.transcripts/compact_0001.json'],
                            'recent_transcript': ['summary: checkpoint-1'],
                            'last_compaction_reason': 'manual',
                        }
                    },
                },
            )

            context = store.load_context(goal='g1', last_k_steps=5)

            compression_state = context.state_summary.get('compression_state', {})
            self.assertEqual(compression_state.get('summary'), 'checkpoint-1')
            self.assertEqual(compression_state.get('compaction_count'), 1)
            self.assertEqual(compression_state.get('last_compaction_reason'), 'manual')
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
