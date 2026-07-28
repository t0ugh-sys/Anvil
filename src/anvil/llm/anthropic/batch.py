from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List

from .._http import ProviderHttpError


@dataclass
class BatchRequest:
    """A single request in a batch."""
    custom_id: str
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.2
    stop_sequences: List[str] = None
    thinking_budget_tokens: int = 0

    def to_anthropic_request(self, model: str) -> Dict[str, object]:
        """Convert to Anthropic batch request format."""
        params: Dict[str, object] = {
            'model': model,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'messages': [{'role': 'user', 'content': self.prompt}],
        }
        if self.stop_sequences:
            params['stop_sequences'] = self.stop_sequences
        if self.thinking_budget_tokens > 0:
            params['thinking'] = {
                'type': 'enabled',
                'budget_tokens': self.thinking_budget_tokens,
            }
            params['temperature'] = 1.0
            params['max_tokens'] = max(self.thinking_budget_tokens + 4096, self.thinking_budget_tokens * 2)
        return {
            'custom_id': self.custom_id,
            'params': params,
        }


@dataclass
class BatchResult:
    """Result from a batch request."""
    custom_id: str
    text: str = ''
    error: str = ''
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def ok(self) -> bool:
        return not self.error


class AnthropicBatchClient:
    """Claude Batch API client for cost-effective bulk processing.

    Batch API provides 50% cost savings on input/output tokens.
    Requests are processed within 24 hours (typically much faster).

    Usage::

        client = AnthropicBatchClient(api_key='...', model='claude-sonnet-5')
        batch_id = client.submit([
            BatchRequest(custom_id='req1', prompt='Analyze this code...'),
            BatchRequest(custom_id='req2', prompt='Review this PR...'),
        ])
        # ... wait ...
        results = client.get_results(batch_id)
    """

    def __init__(
        self,
        api_key: str,
        model: str = 'claude-sonnet-5',
        base_url: str = '',
        timeout_s: float = 30.0,
    ):
        self.api_key = api_key
        self.model = model
        self.base = (base_url.rstrip('/') if base_url else 'https://api.anthropic.com')
        self.timeout_s = timeout_s
        self._headers = {
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
        }

    def submit(self, requests: List[BatchRequest]) -> str:
        """Submit a batch of requests. Returns batch_id."""
        endpoint = self.base + '/v1/messages/batches'
        batch_items = [r.to_anthropic_request(self.model) for r in requests]
        payload = {'requests': batch_items}
        result = self._post(endpoint, payload)
        return result.get('id', '')

    def get_status(self, batch_id: str) -> Dict[str, object]:
        """Check batch status."""
        endpoint = f'{self.base}/v1/messages/batches/{batch_id}'
        return self._get(endpoint)

    def get_results(self, batch_id: str) -> List[BatchResult]:
        """Get results from a completed batch."""
        status = self.get_status(batch_id)
        processing = status.get('processing_status', '')
        if processing != 'ended':
            return []

        results_url = status.get('results_url', '')
        if not results_url:
            return []

        # Fetch results
        raw_results = self._get_raw(self.base + results_url)
        results: List[BatchResult] = []
        for line in raw_results.strip().split('\n'):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                custom_id = item.get('custom_id', '')
                result_type = item.get('type', '')

                if result_type == 'succeeded':
                    msg = item.get('result', {}).get('message', {})
                    content = msg.get('content', [])
                    text_parts = [b.get('text', '') for b in content if isinstance(b, dict) and b.get('type') == 'text']
                    usage = msg.get('usage', {})
                    results.append(BatchResult(
                        custom_id=custom_id,
                        text='\n'.join(text_parts),
                        input_tokens=int(usage.get('input_tokens', 0)),
                        output_tokens=int(usage.get('output_tokens', 0)),
                    ))
                else:
                    error = item.get('error', {})
                    results.append(BatchResult(
                        custom_id=custom_id,
                        error=error.get('message', 'unknown error'),
                    ))
            except (json.JSONDecodeError, KeyError):
                continue

        return results

    def cancel(self, batch_id: str) -> Dict[str, object]:
        """Cancel a batch."""
        endpoint = f'{self.base}/v1/messages/batches/{batch_id}/cancel'
        return self._post(endpoint, {})

    def _post(self, endpoint: str, payload: dict) -> dict:
        body = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(endpoint, data=body, headers=self._headers, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except urllib.error.HTTPError as exc:
            error_body = ''
            try:
                error_body = exc.read().decode('utf-8', errors='replace')
            except Exception:
                error_body = str(exc)
            raise ProviderHttpError(status_code=int(exc.code), body=error_body) from exc

    def _get(self, endpoint: str) -> dict:
        req = urllib.request.Request(endpoint, headers=self._headers, method='GET')
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except urllib.error.HTTPError as exc:
            error_body = ''
            try:
                error_body = exc.read().decode('utf-8', errors='replace')
            except Exception:
                error_body = str(exc)
            raise ProviderHttpError(status_code=int(exc.code), body=error_body) from exc

    def _get_raw(self, endpoint: str) -> str:
        req = urllib.request.Request(endpoint, headers=self._headers, method='GET')
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return resp.read().decode('utf-8')
        except urllib.error.HTTPError as exc:
            error_body = ''
            try:
                error_body = exc.read().decode('utf-8', errors='replace')
            except Exception:
                error_body = str(exc)
            raise ProviderHttpError(status_code=int(exc.code), body=error_body) from exc
