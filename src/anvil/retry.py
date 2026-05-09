"""
Retry logic with exponential backoff and jitter.

Inspired by Claude Code's withRetry pattern:
- Exponential backoff with configurable base and cap
- 25% jitter to prevent thundering herd
- Retry-After header respect
- Overload (529) specific handling with consecutive failure tracking
- Auth error fast-fail (no retry on 401/403)
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Set, TypeVar

T = TypeVar('T')

# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_BACKOFF_S = 0.5
DEFAULT_MAX_BACKOFF_S = 32.0
DEFAULT_JITTER_FACTOR = 0.25

# HTTP codes that should be retried (transient errors)
RETRYABLE_HTTP_CODES: frozenset[int] = frozenset({429, 502, 503, 504, 524, 529})

# HTTP codes that should NEVER be retried (auth/client errors)
NON_RETRYABLE_HTTP_CODES: frozenset[int] = frozenset({401, 403, 404, 422})

# 529 = overloaded, needs special handling
OVERLOAD_HTTP_CODE = 529
MAX_CONSECUTIVE_OVERLOADS = 3


class RetryExhausted(Exception):
    """All retry attempts exhausted."""

    def __init__(self, last_status_code: int, last_body: str, attempts: int):
        self.last_status_code = last_status_code
        self.last_body = last_body
        self.attempts = attempts
        super().__init__(
            f'Retry exhausted after {attempts} attempts (last: HTTP {last_status_code})'
        )


class NonRetryableError(Exception):
    """Error that should not be retried (auth, validation, etc.)."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f'HTTP {status_code}: {body}')


class OverloadError(Exception):
    """Consecutive 529 overloads exceeded threshold."""

    def __init__(self, consecutive: int):
        self.consecutive = consecutive
        super().__init__(f'Consecutive overloads: {consecutive}')


@dataclass
class RetryState:
    """Track retry state across calls."""
    attempt: int = 0
    consecutive_overloads: int = 0
    last_retry_after: float | None = None

    def record_success(self) -> None:
        self.consecutive_overloads = 0
        self.last_retry_after = None

    def record_overload(self, retry_after: float | None = None) -> None:
        self.consecutive_overloads += 1
        if retry_after is not None:
            self.last_retry_after = retry_after

    def record_retryable(self) -> None:
        self.consecutive_overloads = 0


def compute_backoff(
    attempt: int,
    base_s: float = DEFAULT_BASE_BACKOFF_S,
    max_s: float = DEFAULT_MAX_BACKOFF_S,
    jitter_factor: float = DEFAULT_JITTER_FACTOR,
) -> float:
    """Compute backoff with exponential growth and jitter.

    Returns seconds to sleep before next retry.
    """
    exponential = base_s * (2 ** attempt)
    capped = min(exponential, max_s)
    jitter = capped * jitter_factor * random.random()
    return capped + jitter


def parse_retry_after(headers: dict[str, str]) -> float | None:
    """Parse Retry-After header value (seconds)."""
    value = headers.get('retry-after') or headers.get('Retry-After')
    if value is None:
        return None
    try:
        seconds = float(value)
        return max(0.0, seconds)
    except (ValueError, TypeError):
        return None


def with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_backoff_s: float = DEFAULT_BASE_BACKOFF_S,
    max_backoff_s: float = DEFAULT_MAX_BACKOFF_S,
    retryable_codes: Set[int] | None = None,
    state: RetryState | None = None,
    on_retry: Callable[[int, float], None] | None = None,
    get_status_code: Callable[[Exception], int | None],
    get_retry_after: Callable[[Exception], float | None] = lambda _: None,
    get_body: Callable[[Exception], str] = lambda _: '',
) -> T:
    """Execute fn with retry logic.

    Args:
        fn: Function to execute
        max_retries: Maximum retry attempts
        base_backoff_s: Base backoff duration
        max_backoff_s: Maximum backoff duration
        retryable_codes: HTTP codes to retry (default: RETRYABLE_HTTP_CODES)
        state: Shared retry state for overload tracking
        on_retry: Callback(attempt, sleep_seconds) before each retry
        get_status_code: Extract HTTP status from exception
        get_retry_after: Extract Retry-After from exception
        get_body: Extract response body from exception

    Returns:
        Result of fn()

    Raises:
        RetryExhausted: All retries used
        NonRetryableError: Non-retryable HTTP error
        OverloadError: Too many consecutive 529s
    """
    if retryable_codes is None:
        retryable_codes = set(RETRYABLE_HTTP_CODES)

    if state is None:
        state = RetryState()

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        state.attempt = attempt
        try:
            result = fn()
            state.record_success()
            return result
        except Exception as exc:
            last_error = exc
            status = get_status_code(exc)
            body = get_body(exc)

            # Non-retryable: fail immediately
            if status is not None and status in NON_RETRYABLE_HTTP_CODES:
                raise NonRetryableError(status, body) from exc

            # 529 overload: track consecutive failures
            is_overload = status == OVERLOAD_HTTP_CODE
            if is_overload:
                state.record_overload(get_retry_after(exc))
                if state.consecutive_overloads >= MAX_CONSECUTIVE_OVERLOADS:
                    raise OverloadError(state.consecutive_overloads) from exc

            # Not retryable or exhausted
            if status is None or status not in retryable_codes or attempt >= max_retries:
                if attempt >= max_retries:
                    raise RetryExhausted(
                        last_status_code=status or 0,
                        last_body=body,
                        attempts=attempt + 1,
                    ) from exc
                raise

            # Compute backoff
            retry_after = get_retry_after(exc)
            if retry_after is not None:
                sleep_s = retry_after
            else:
                sleep_s = compute_backoff(attempt, base_backoff_s, max_backoff_s)

            # Don't reset overload counter for 529 retries
            if not is_overload:
                state.record_retryable()
            if on_retry:
                on_retry(attempt, sleep_s)
            time.sleep(sleep_s)

    # Should not reach here, but just in case
    if last_error is not None:
        raise last_error
    raise ValueError('retry loop completed without result or error')
