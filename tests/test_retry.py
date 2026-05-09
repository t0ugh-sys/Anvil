from __future__ import annotations

import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401

from anvil.retry import (
    DEFAULT_BASE_BACKOFF_S,
    DEFAULT_MAX_BACKOFF_S,
    NonRetryableError,
    OverloadError,
    RetryExhausted,
    RetryState,
    compute_backoff,
    parse_retry_after,
    with_retry,
)


class ComputeBackoffTests(unittest.TestCase):
    def test_should_grow_exponentially(self) -> None:
        b0 = compute_backoff(0, base_s=1.0, jitter_factor=0)
        b1 = compute_backoff(1, base_s=1.0, jitter_factor=0)
        b2 = compute_backoff(2, base_s=1.0, jitter_factor=0)
        self.assertAlmostEqual(b0, 1.0)
        self.assertAlmostEqual(b1, 2.0)
        self.assertAlmostEqual(b2, 4.0)

    def test_should_cap_at_max(self) -> None:
        b = compute_backoff(20, base_s=0.5, max_s=32.0, jitter_factor=0)
        self.assertEqual(b, 32.0)

    def test_should_add_jitter(self) -> None:
        results = {compute_backoff(1, base_s=1.0, jitter_factor=0.25) for _ in range(100)}
        # With jitter, we should get varied results
        self.assertGreater(len(results), 1)


class ParseRetryAfterTests(unittest.TestCase):
    def test_should_parse_seconds(self) -> None:
        self.assertEqual(parse_retry_after({'retry-after': '5'}), 5.0)

    def test_should_parse_case_insensitive(self) -> None:
        self.assertEqual(parse_retry_after({'Retry-After': '3'}), 3.0)

    def test_should_return_none_if_missing(self) -> None:
        self.assertIsNone(parse_retry_after({}))

    def test_should_return_none_for_invalid(self) -> None:
        self.assertIsNone(parse_retry_after({'retry-after': 'abc'}))


class RetryStateTests(unittest.TestCase):
    def test_should_track_consecutive_overloads(self) -> None:
        state = RetryState()
        self.assertEqual(state.consecutive_overloads, 0)
        state.record_overload()
        self.assertEqual(state.consecutive_overloads, 1)
        state.record_overload()
        self.assertEqual(state.consecutive_overloads, 2)

    def test_should_reset_on_success(self) -> None:
        state = RetryState()
        state.record_overload()
        state.record_overload()
        state.record_success()
        self.assertEqual(state.consecutive_overloads, 0)

    def test_should_reset_overloads_on_retryable(self) -> None:
        state = RetryState()
        state.record_overload()
        state.record_retryable()
        self.assertEqual(state.consecutive_overloads, 0)


class WithRetryTests(unittest.TestCase):
    def test_should_return_on_first_success(self) -> None:
        call_count = [0]
        def fn():
            call_count[0] += 1
            return 'ok'

        result = with_retry(fn, get_status_code=lambda e: None)
        self.assertEqual(result, 'ok')
        self.assertEqual(call_count[0], 1)

    def test_should_retry_on_retryable_error(self) -> None:
        call_count = [0]
        class FakeError(Exception):
            status_code = 503
        def fn():
            call_count[0] += 1
            if call_count[0] < 3:
                raise FakeError('service unavailable')
            return 'ok'

        result = with_retry(
            fn,
            max_retries=3,
            base_backoff_s=0.01,
            get_status_code=lambda e: getattr(e, 'status_code', None),
        )
        self.assertEqual(result, 'ok')
        self.assertEqual(call_count[0], 3)

    def test_should_fail_on_non_retryable_error(self) -> None:
        class AuthError(Exception):
            status_code = 401
        def fn():
            raise AuthError('unauthorized')

        with self.assertRaises(NonRetryableError) as ctx:
            with_retry(
                fn,
                max_retries=3,
                get_status_code=lambda e: getattr(e, 'status_code', None),
            )
        self.assertEqual(ctx.exception.status_code, 401)

    def test_should_raise_retry_exhausted(self) -> None:
        class ServerError(Exception):
            status_code = 503
        def fn():
            raise ServerError('down')

        with self.assertRaises(RetryExhausted) as ctx:
            with_retry(
                fn,
                max_retries=2,
                base_backoff_s=0.01,
                get_status_code=lambda e: getattr(e, 'status_code', None),
            )
        self.assertEqual(ctx.exception.attempts, 3)

    def test_should_raise_overload_after_consecutive_529s(self) -> None:
        class OverloadError529(Exception):
            status_code = 529
        def fn():
            raise OverloadError529('overloaded')

        with self.assertRaises(OverloadError):
            with_retry(
                fn,
                max_retries=10,
                base_backoff_s=0.01,
                get_status_code=lambda e: getattr(e, 'status_code', None),
            )

    def test_should_use_retry_after_header(self) -> None:
        sleep_calls = []
        call_count = [0]
        class RateLimitError(Exception):
            status_code = 429
            retry_after = 2.0
        def fn():
            call_count[0] += 1
            if call_count[0] < 2:
                raise RateLimitError('rate limited')
            return 'ok'

        def on_retry(attempt, sleep_s):
            sleep_calls.append(sleep_s)

        result = with_retry(
            fn,
            max_retries=3,
            base_backoff_s=0.01,
            get_status_code=lambda e: getattr(e, 'status_code', None),
            get_retry_after=lambda e: getattr(e, 'retry_after', None),
            on_retry=on_retry,
        )
        self.assertEqual(result, 'ok')
        self.assertEqual(len(sleep_calls), 1)
        self.assertEqual(sleep_calls[0], 2.0)

    def test_should_call_on_retry_callback(self) -> None:
        retry_log = []
        call_count = [0]
        class ServerError(Exception):
            status_code = 503
        def fn():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ServerError('down')
            return 'ok'

        def on_retry(attempt, sleep_s):
            retry_log.append((attempt, sleep_s))

        with_retry(
            fn,
            max_retries=3,
            base_backoff_s=0.01,
            get_status_code=lambda e: getattr(e, 'status_code', None),
            on_retry=on_retry,
        )
        self.assertEqual(len(retry_log), 1)
        self.assertEqual(retry_log[0][0], 0)


if __name__ == '__main__':
    unittest.main()
