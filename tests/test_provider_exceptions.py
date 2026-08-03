from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock
import urllib.error

import _bootstrap  # noqa: F401

from anvil.errors import (
    AuthError,
    ModelNotFoundError,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
    RateLimitError,
)
from anvil.llm._http import ProviderHttpError, _map_http_error, _http_post_json


class MapHttpErrorTests(unittest.TestCase):
    """Test _map_http_error() mapping table for all status code branches."""

    def test_should_map_429_to_rate_limit_error(self):
        exc = ProviderHttpError(status_code=429, body='rate limited', retry_after=30.0)
        result = _map_http_error(exc)
        self.assertIsInstance(result, RateLimitError)
        self.assertEqual(result.retry_after, 30.0)

    def test_should_map_401_to_auth_error(self):
        exc = ProviderHttpError(status_code=401, body='unauthorized')
        result = _map_http_error(exc)
        self.assertIsInstance(result, AuthError)

    def test_should_map_403_to_auth_error(self):
        exc = ProviderHttpError(status_code=403, body='forbidden')
        result = _map_http_error(exc)
        self.assertIsInstance(result, AuthError)

    def test_should_map_404_to_model_not_found_error(self):
        exc = ProviderHttpError(status_code=404, body='model not found')
        result = _map_http_error(exc)
        self.assertIsInstance(result, ModelNotFoundError)

    def test_should_map_other_status_to_provider_response_error(self):
        exc = ProviderHttpError(status_code=500, body='internal server error')
        result = _map_http_error(exc)
        self.assertIsInstance(result, ProviderResponseError)
        self.assertEqual(result.status_code, 500)


class RetryAfterHeaderTests(unittest.TestCase):
    """Test that Retry-After header is parsed and propagated."""

    def test_should_parse_retry_after_header_from_http_error(self):
        """Test that _http_post_json extracts Retry-After from HTTPError.headers."""
        mock_response = MagicMock()
        mock_http_error = urllib.error.HTTPError(
            url='https://api.example.com/v1/test',
            code=429,
            msg='Too Many Requests',
            hdrs={'Retry-After': '42'},
            fp=None,
        )
        mock_http_error.read = MagicMock(return_value=b'rate limit exceeded')

        with patch('urllib.request.urlopen', side_effect=mock_http_error):
            with self.assertRaises(ProviderHttpError) as cm:
                _http_post_json(
                    endpoint='https://api.example.com/v1/test',
                    payload={'test': 'data'},
                    headers={'Content-Type': 'application/json'},
                    timeout_s=10.0,
                )
            self.assertEqual(cm.exception.status_code, 429)
            self.assertEqual(cm.exception.retry_after, 42.0)

    def test_should_handle_missing_retry_after_header(self):
        """Test that missing Retry-After header results in None."""
        mock_http_error = urllib.error.HTTPError(
            url='https://api.example.com/v1/test',
            code=429,
            msg='Too Many Requests',
            hdrs={},
            fp=None,
        )
        mock_http_error.read = MagicMock(return_value=b'rate limit exceeded')

        with patch('urllib.request.urlopen', side_effect=mock_http_error):
            with self.assertRaises(ProviderHttpError) as cm:
                _http_post_json(
                    endpoint='https://api.example.com/v1/test',
                    payload={'test': 'data'},
                    headers={'Content-Type': 'application/json'},
                    timeout_s=10.0,
                )
            self.assertIsNone(cm.exception.retry_after)


class URLErrorHandlingTests(unittest.TestCase):
    """Test that URLError (network timeout) is mapped to ProviderTimeoutError."""

    def test_should_map_url_error_to_provider_timeout_error(self):
        """Test that urllib.error.URLError raises ProviderTimeoutError."""
        mock_url_error = urllib.error.URLError('Connection timed out')

        with patch('urllib.request.urlopen', side_effect=mock_url_error):
            with self.assertRaises(ProviderTimeoutError) as cm:
                _http_post_json(
                    endpoint='https://api.example.com/v1/test',
                    payload={'test': 'data'},
                    headers={'Content-Type': 'application/json'},
                    timeout_s=10.0,
                )
            self.assertIn('Connection timed out', str(cm.exception))


class ProviderErrorMROTests(unittest.TestCase):
    """Test that ProviderError IS-A ValueError for backward compatibility."""

    def test_provider_error_is_a_value_error(self):
        """ProviderError(AnvilError, ValueError) should pass isinstance(ValueError)."""
        exc = ProviderError('test error')
        self.assertIsInstance(exc, ValueError)

    def test_rate_limit_error_is_a_value_error(self):
        exc = RateLimitError('rate limited', retry_after=10.0)
        self.assertIsInstance(exc, ValueError)

    def test_auth_error_is_a_value_error(self):
        exc = AuthError('unauthorized')
        self.assertIsInstance(exc, ValueError)

    def test_model_not_found_error_is_a_value_error(self):
        exc = ModelNotFoundError('claude-99')
        self.assertIsInstance(exc, ValueError)

    def test_provider_timeout_error_is_a_value_error(self):
        exc = ProviderTimeoutError('timeout')
        self.assertIsInstance(exc, ValueError)

    def test_provider_response_error_is_a_value_error(self):
        exc = ProviderResponseError('bad format', status_code=502)
        self.assertIsInstance(exc, ValueError)


if __name__ == '__main__':
    unittest.main()
