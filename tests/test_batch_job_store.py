from __future__ import annotations

import shutil
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401

from anvil.llm.anthropic.batch import BatchJobStore


class BatchJobStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path('tests/.tmp') / 'batch-job-store'
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.tmp_dir / 'batch_jobs.sqlite3'

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_should_save_and_load_pending_job(self) -> None:
        store = BatchJobStore(self.db_path)
        store.save('batch_1', 'claude-sonnet-5', {'label': 'review'})

        pending = store.load_pending()

        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].batch_id, 'batch_1')
        self.assertEqual(pending[0].model, 'claude-sonnet-5')
        self.assertEqual(pending[0].status, 'pending')
        self.assertEqual(pending[0].metadata, {'label': 'review'})

    def test_should_exclude_done_jobs_from_pending(self) -> None:
        store = BatchJobStore(self.db_path)
        store.save('batch_1', 'claude-sonnet-5', {})
        store.mark_done('batch_1')

        self.assertEqual(store.load_pending(), [])

    def test_should_mark_job_cancelled(self) -> None:
        store = BatchJobStore(self.db_path)
        store.save('batch_1', 'claude-sonnet-5', {})
        store.mark_done('batch_1', status='cancelled')

        self.assertEqual(store.load_pending(), [])

    def test_should_persist_across_store_instances(self) -> None:
        BatchJobStore(self.db_path).save('batch_1', 'claude-sonnet-5', {})

        reopened = BatchJobStore(self.db_path)
        pending = reopened.load_pending()

        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].batch_id, 'batch_1')

    def test_should_replace_job_on_duplicate_save(self) -> None:
        store = BatchJobStore(self.db_path)
        store.save('batch_1', 'claude-sonnet-5', {'v': 1})
        store.save('batch_1', 'claude-opus-5', {'v': 2})

        pending = store.load_pending()

        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].model, 'claude-opus-5')
        self.assertEqual(pending[0].metadata, {'v': 2})


if __name__ == '__main__':
    unittest.main()
