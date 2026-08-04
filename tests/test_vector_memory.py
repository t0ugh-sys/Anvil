from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401

from anvil.memory.vector_store import (
    VectorMemoryStore,
    MemoryEntry,
    MemoryResult,
    _default_embed,
    _cosine,
)


def _tmp_db() -> Path:
    return Path('tests/.tmp') / f'vmem-{uuid.uuid4().hex}' / 'vectors.db'


class TestDefaultEmbed(unittest.TestCase):
    def test_returns_unit_vector(self):
        vec = _default_embed('hello world')
        norm = sum(x * x for x in vec) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_empty_string_returns_zero_vector(self):
        vec = _default_embed('')
        self.assertEqual(sum(vec), 0.0)

    def test_different_texts_produce_different_vectors(self):
        a = _default_embed('python asyncio')
        b = _default_embed('sqlite database')
        self.assertNotEqual(a, b)

    def test_same_text_produces_same_vector(self):
        a = _default_embed('reproducible embedding')
        b = _default_embed('reproducible embedding')
        self.assertEqual(a, b)


class TestVectorMemoryStore(unittest.TestCase):
    def setUp(self):
        self.db_path = _tmp_db()

    def tearDown(self):
        shutil.rmtree(self.db_path.parent.parent, ignore_errors=True)

    def test_add_and_get(self):
        store = VectorMemoryStore(self.db_path)
        store.add('k1', 'asyncio event loop', {'tag': 'async'})
        entry = store.get('k1')
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.key, 'k1')
        self.assertEqual(entry.text, 'asyncio event loop')
        self.assertEqual(entry.metadata['tag'], 'async')

    def test_get_missing_returns_none(self):
        store = VectorMemoryStore(self.db_path)
        self.assertIsNone(store.get('nonexistent'))

    def test_delete_removes_entry(self):
        store = VectorMemoryStore(self.db_path)
        store.add('k1', 'to be deleted')
        store.delete('k1')
        self.assertIsNone(store.get('k1'))
        self.assertEqual(store.count(), 0)

    def test_count(self):
        store = VectorMemoryStore(self.db_path)
        self.assertEqual(store.count(), 0)
        store.add('a', 'first entry')
        store.add('b', 'second entry')
        self.assertEqual(store.count(), 2)

    def test_replace_on_duplicate_key(self):
        store = VectorMemoryStore(self.db_path)
        store.add('k1', 'original text')
        store.add('k1', 'updated text')
        self.assertEqual(store.count(), 1)
        entry = store.get('k1')
        assert entry is not None
        self.assertEqual(entry.text, 'updated text')

    def test_search_returns_top_k(self):
        store = VectorMemoryStore(self.db_path)
        for i in range(10):
            store.add(f'entry-{i}', f'document number {i} about python')
        results = store.search('python', top_k=3)
        self.assertEqual(len(results), 3)

    def test_search_relevance_ordering(self):
        store = VectorMemoryStore(self.db_path)
        store.add('async', 'asyncio gather await coroutine event loop')
        store.add('sql', 'sqlite database table insert select query')
        store.add('net', 'network socket tcp udp connection')

        results = store.search('asyncio coroutine')
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].entry.key, 'async')

    def test_search_empty_store_returns_empty(self):
        store = VectorMemoryStore(self.db_path)
        results = store.search('anything')
        self.assertEqual(results, [])

    def test_all_entries(self):
        store = VectorMemoryStore(self.db_path)
        store.add('a', 'first')
        store.add('b', 'second')
        entries = store.all_entries()
        self.assertEqual(len(entries), 2)
        keys = {e.key for e in entries}
        self.assertEqual(keys, {'a', 'b'})

    def test_tier_isolation(self):
        store_long = VectorMemoryStore(self.db_path, tier='long')
        store_proj = VectorMemoryStore(self.db_path, tier='project')
        store_long.add('k1', 'long-term memory')
        store_proj.add('k2', 'project memory')

        self.assertEqual(store_long.count(), 1)
        self.assertEqual(store_proj.count(), 1)
        self.assertEqual(store_long.count(tier='project'), 1)
        long_results = store_long.search('memory', tier='long')
        proj_results = store_long.search('memory', tier='project')
        self.assertEqual(long_results[0].entry.key, 'k1')
        self.assertEqual(proj_results[0].entry.key, 'k2')

    def test_inject_into_prompt_empty(self):
        store = VectorMemoryStore(self.db_path)
        self.assertEqual(store.inject_into_prompt('query'), '')

    def test_inject_into_prompt_format(self):
        store = VectorMemoryStore(self.db_path)
        store.add('tip1', 'use asyncio.gather for parallel tasks')
        result = store.inject_into_prompt('asyncio')
        self.assertIn('Relevant memories from previous sessions:', result)
        self.assertIn('[tip1]', result)
        self.assertIn('asyncio.gather', result)

    def test_inject_into_prompt_top_k(self):
        store = VectorMemoryStore(self.db_path)
        for i in range(10):
            store.add(f'k{i}', f'memory entry {i} about testing')
        result = store.inject_into_prompt('testing', top_k=3)
        lines = result.strip().split('\n')
        # header + 3 entries
        self.assertEqual(len(lines), 4)

    def test_custom_embed_fn(self):
        calls: list[str] = []

        def tracking_embed(text: str) -> list[float]:
            calls.append(text)
            return _default_embed(text)

        store = VectorMemoryStore(self.db_path, embed_fn=tracking_embed)
        store.add('k1', 'some text')
        store.search('query')
        self.assertIn('some text', calls)
        self.assertIn('query', calls)

    def test_result_score_is_float(self):
        store = VectorMemoryStore(self.db_path)
        store.add('k1', 'test entry')
        results = store.search('test')
        self.assertIsInstance(results[0].score, float)


if __name__ == '__main__':
    unittest.main()
