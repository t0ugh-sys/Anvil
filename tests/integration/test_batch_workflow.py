"""Integration: BatchRequest/BatchResult dataclasses and AnthropicBatchClient with mocked HTTP."""
from __future__ import annotations

import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import _bootstrap  # noqa: F401

import unittest
from unittest.mock import patch, MagicMock

from anvil.llm.anthropic.batch import BatchRequest, BatchResult, AnthropicBatchClient


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


if __name__ == '__main__':
    unittest.main()
