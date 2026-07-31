from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import _bootstrap  # noqa: F401

from anvil.errors import ProviderError
from anvil.llm.anthropic.client import (
    _anthropic_async_invoke_factory,
    anthropic_async_invoke_factory,
)
from anvil.llm.usage import TokenUsageTracker


def _run(coro):
    return asyncio.run(coro)


class AnthropicAsyncInvokeFactoryTests(unittest.TestCase):
    def test_should_return_text_from_successful_response(self) -> None:
        invoke = anthropic_async_invoke_factory(api_key='test-key', model='claude-sonnet-5')
        response = {'content': [{'type': 'text', 'text': 'hello async'}]}
        with patch(
            'anvil.llm.anthropic.client._http_post_json_async',
            new=AsyncMock(return_value=response),
        ):
            result = _run(invoke('hi'))
        self.assertEqual(result, 'hello async')

    def test_should_record_usage_when_tracker_provided(self) -> None:
        tracker = TokenUsageTracker()
        invoke = anthropic_async_invoke_factory(
            api_key='test-key', model='claude-sonnet-5', usage_tracker=tracker,
        )
        response = {
            'content': [{'type': 'text', 'text': 'ok'}],
            'usage': {'input_tokens': 10, 'output_tokens': 5},
        }
        with patch(
            'anvil.llm.anthropic.client._http_post_json_async',
            new=AsyncMock(return_value=response),
        ):
            _run(invoke('hi'))
        self.assertEqual(tracker.total_input_tokens, 10)
        self.assertEqual(tracker.total_output_tokens, 5)

    def test_should_retry_on_retryable_status_then_succeed(self) -> None:
        from anvil.llm._http import ProviderHttpError

        invoke = _anthropic_async_invoke_factory(
            api_key='test-key',
            model='claude-sonnet-5',
            temperature=0.2,
            timeout_s=10.0,
            max_retries=2,
            retry_backoff_s=0.001,
            retry_http_codes={503},
        )
        success_response = {'content': [{'type': 'text', 'text': 'recovered'}]}
        mock_call = AsyncMock(
            side_effect=[
                ProviderHttpError(status_code=503, body='overloaded'),
                success_response,
            ]
        )
        with patch('anvil.llm.anthropic.client._http_post_json_async', new=mock_call):
            result = _run(invoke('hi'))
        self.assertEqual(result, 'recovered')
        self.assertEqual(mock_call.await_count, 2)

    def test_should_raise_value_error_on_non_retryable_status(self) -> None:
        from anvil.llm._http import ProviderHttpError

        invoke = _anthropic_async_invoke_factory(
            api_key='test-key',
            model='claude-sonnet-5',
            temperature=0.2,
            timeout_s=10.0,
            max_retries=1,
            retry_backoff_s=0.001,
            retry_http_codes={503},
        )
        mock_call = AsyncMock(side_effect=ProviderHttpError(status_code=401, body='bad key'))
        with patch('anvil.llm.anthropic.client._http_post_json_async', new=mock_call):
            with self.assertRaises(ProviderError):
                _run(invoke('hi'))
        self.assertEqual(mock_call.await_count, 1)

    def test_should_exhaust_retries_and_raise(self) -> None:
        from anvil.llm._http import ProviderHttpError

        invoke = _anthropic_async_invoke_factory(
            api_key='test-key',
            model='claude-sonnet-5',
            temperature=0.2,
            timeout_s=10.0,
            max_retries=2,
            retry_backoff_s=0.001,
            retry_http_codes={503},
        )
        mock_call = AsyncMock(side_effect=ProviderHttpError(status_code=503, body='down'))
        with patch('anvil.llm.anthropic.client._http_post_json_async', new=mock_call):
            with self.assertRaises(ProviderError):
                _run(invoke('hi'))
        self.assertEqual(mock_call.await_count, 3)


class HttpPostJsonAsyncTests(unittest.TestCase):
    def test_should_delegate_to_sync_helper_via_thread(self) -> None:
        from anvil.llm._http import _http_post_json_async

        with patch('anvil.llm._http._http_post_json', return_value={'ok': True}) as mock_sync:
            result = _run(_http_post_json_async('https://example.com', {}, {}, 5.0))
        self.assertEqual(result, {'ok': True})
        mock_sync.assert_called_once()


if __name__ == '__main__':
    unittest.main()
