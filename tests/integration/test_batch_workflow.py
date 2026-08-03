"""Integration: BatchRequest/BatchResult dataclasses and AnthropicBatchClient with mocked HTTP."""
from __future__ import annotations

import json
import shutil
import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import _bootstrap  # noqa: F401

import unittest
from unittest.mock import patch, MagicMock

from anvil.llm.anthropic.batch import BatchRequest, BatchResult, BatchJobStore, AnthropicBatchClient


class TestBatchDataclasses(unittest.TestCase):
    def test_batch_request_to_anthropic_format(self):
        req = BatchRequest(custom_id='r1', prompt='hello', max_tokens=512)
        payload = req.to_anthropic_request('claude-sonnet-5')
        self.assertEqual(payload['custom_id'], 'r1')
        self.assertEqual(payload['params']['model'], 'claude-sonnet-5')
        self.assertEqual(payload['params']['messages'][0]['content'], 'hello')

    def test_batch_result_ok_property(self):
        ok = BatchResult(custom_id='r1', text='done')
        self.assertTrue(ok.ok)
        err = BatchResult(custom_id='r2', error='rate limited')
        self.assertFalse(err.ok)

    def test_thinking_budget_overrides_temperature(self):
        req = BatchRequest(custom_id='r1', prompt='think', thinking_budget_tokens=1000)
        payload = req.to_anthropic_request('claude-sonnet-5')
        self.assertEqual(payload['params']['temperature'], 1.0)
        self.assertIn('thinking', payload['params'])


class TestBatchClientWorkflow(unittest.TestCase):
    def _make_client(self) -> AnthropicBatchClient:
        return AnthropicBatchClient(api_key='test-key', model='claude-sonnet-5')

    def test_submit_returns_batch_id(self):
        client = self._make_client()
        with patch.object(client, '_post', return_value={'id': 'batch_abc123'}) as mock_post:
            batch_id = client.submit([BatchRequest(custom_id='r1', prompt='hello')])
            self.assertEqual(batch_id, 'batch_abc123')
            mock_post.assert_called_once()

    def test_get_status_returns_dict(self):
        client = self._make_client()
        status_payload = {'id': 'batch_abc123', 'processing_status': 'in_progress'}
        with patch.object(client, '_get', return_value=status_payload):
            status = client.get_status('batch_abc123')
            self.assertEqual(status['processing_status'], 'in_progress')

    def test_get_results_empty_when_not_ended(self):
        client = self._make_client()
        status_payload = {'id': 'batch_abc123', 'processing_status': 'in_progress'}
        with patch.object(client, '_get', return_value=status_payload):
            results = client.get_results('batch_abc123')
            self.assertEqual(results, [])

    def test_full_submit_poll_results_workflow(self):
        client = self._make_client()
        jsonl_body = '{"custom_id": "r1", "type": "succeeded", "result": {"message": {"content": [{"type": "text", "text": "answer1"}], "usage": {"input_tokens": 10, "output_tokens": 5}}}}'
        status_ended = {'id': 'batch_abc', 'processing_status': 'ended',
                        'results_url': '/v1/messages/batches/batch_abc/results'}
        with patch.object(client, '_post', return_value={'id': 'batch_abc'}):
            batch_id = client.submit([BatchRequest(custom_id='r1', prompt='analyze')])
        with patch.object(client, '_get', return_value=status_ended), \
             patch.object(client, '_get_raw', return_value=jsonl_body):
            results = client.get_results(batch_id)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].ok)
            self.assertEqual(results[0].text, 'answer1')


class TestBatchClientWithStore(unittest.TestCase):
    def _make_client_with_store(self) -> tuple[AnthropicBatchClient, BatchJobStore, Path]:
        tmp_dir = Path('tests/.tmp') / 'batch-client-store'
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        store = BatchJobStore(tmp_dir / 'batch_jobs.db')
        client = AnthropicBatchClient(api_key='test-key', model='claude-sonnet-5', store=store)
        return client, store, tmp_dir

    def test_submit_persists_job_to_store(self):
        client, store, tmp_dir = self._make_client_with_store()
        try:
            with patch.object(client, '_post', return_value={'id': 'batch_abc123'}):
                batch_id = client.submit([BatchRequest(custom_id='r1', prompt='hello')], metadata={'run': 1})
            pending = store.load_pending()
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].batch_id, batch_id)
            self.assertEqual(pending[0].metadata, {'run': 1})
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_get_results_marks_job_done_in_store(self):
        client, store, tmp_dir = self._make_client_with_store()
        try:
            with patch.object(client, '_post', return_value={'id': 'batch_abc'}):
                batch_id = client.submit([BatchRequest(custom_id='r1', prompt='hello')])
            status_ended = {
                'id': batch_id,
                'processing_status': 'ended',
                'results_url': f'/v1/messages/batches/{batch_id}/results',
            }
            jsonl_body = (
                '{"custom_id": "r1", "type": "succeeded", '
                '"result": {"message": {"content": [{"type": "text", "text": "answer1"}], '
                '"usage": {"input_tokens": 10, "output_tokens": 5}}}}'
            )
            with patch.object(client, '_get', return_value=status_ended), \
                 patch.object(client, '_get_raw', return_value=jsonl_body):
                client.get_results(batch_id)
            self.assertEqual(store.load_pending(), [])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_cancel_marks_job_cancelled_in_store(self):
        client, store, tmp_dir = self._make_client_with_store()
        try:
            with patch.object(client, '_post', return_value={'id': 'batch_abc'}):
                batch_id = client.submit([BatchRequest(custom_id='r1', prompt='hello')])
            with patch.object(client, '_post', return_value={'status': 'cancelled'}):
                client.cancel(batch_id)
            self.assertEqual(store.load_pending(), [])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_resume_pending_jobs_delegates_to_store(self):
        client, store, tmp_dir = self._make_client_with_store()
        try:
            with patch.object(client, '_post', return_value={'id': 'batch_abc'}):
                client.submit([BatchRequest(custom_id='r1', prompt='hello')])
            resumed = client.resume_pending_jobs()
            self.assertEqual(len(resumed), 1)
            self.assertEqual(resumed[0].batch_id, 'batch_abc')
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_resume_pending_jobs_returns_empty_without_store(self):
        client = AnthropicBatchClient(api_key='test-key', model='claude-sonnet-5')
        self.assertEqual(client.resume_pending_jobs(), [])


if __name__ == '__main__':
    unittest.main()
