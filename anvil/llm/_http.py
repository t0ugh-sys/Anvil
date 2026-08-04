from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable, Dict, Optional, Set

from ..errors import (
    AuthError,
    ModelNotFoundError,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
    RateLimitError,
)
from ..infra.retry import NonRetryableError, RetryExhausted, with_retry
from ._types import DEFAULT_RETRY_HTTP_CODES


class ProviderHttpError(Exception):
    """Internal HTTP transport error — mapped to ProviderError subclasses before leaving this module."""

    def __init__(self, status_code: int, body: str, retry_after: Optional[float] = None) -> None:
        super().__init__(f'HTTP {status_code}: {body}')
        self.status_code = status_code
        self.body = body
        self.retry_after = retry_after


def _map_http_error(exc: ProviderHttpError) -> ProviderError:
    """Map a transport-level HTTP error to the appropriate semantic ProviderError subclass."""
    if exc.status_code == 429:
        return RateLimitError(retry_after=exc.retry_after)
    if exc.status_code in (401, 403):
        return AuthError()
    if exc.status_code == 404:
        return ModelNotFoundError()
    return ProviderResponseError(status_code=exc.status_code)


def _request_with_retry(
    request_fn: Callable[[], dict],
    max_retries: int,
    retry_backoff_s: float,
    retry_http_codes: Set[int],
) -> dict:
    """Execute HTTP request with exponential backoff retry."""
    try:
        return with_retry(
            request_fn,
            max_retries=max_retries,
            base_backoff_s=retry_backoff_s,
            retryable_codes=retry_http_codes,
            get_status_code=lambda e: getattr(e, 'status_code', None),
            get_body=lambda e: getattr(e, 'body', ''),
            get_retry_after=lambda e: getattr(e, 'retry_after', None),
        )
    except (RetryExhausted, NonRetryableError) as exc:
        status = getattr(exc, 'last_status_code', None) or getattr(exc, 'status_code', 0)
        body = getattr(exc, 'last_body', None) or getattr(exc, 'body', '')
        retry_after = getattr(exc, 'retry_after', None)
        http_exc = ProviderHttpError(status_code=status, body=body, retry_after=retry_after)
        raise _map_http_error(http_exc) from exc


def _http_post_json(
    endpoint: str,
    payload: dict,
    headers: Dict[str, str],
    timeout_s: float,
    *,
    return_headers: bool = False,
) -> dict | tuple[dict, Dict[str, str]]:
    """Shared HTTP POST helper — serialises payload, sends request, returns parsed JSON."""
    body = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(endpoint, data=body, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode('utf-8')
            if not raw.strip():
                raise ProviderHttpError(status_code=200, body='empty response body')
            parsed = json.loads(raw)
            if return_headers:
                response_headers = dict(response.headers)
                return parsed, response_headers
            return parsed
    except urllib.error.HTTPError as exc:
        error_body = ''
        try:
            error_body = exc.read().decode('utf-8', errors='replace')
        except Exception:
            error_body = str(exc)
        retry_after: Optional[float] = None
        try:
            raw_ra = exc.headers.get('Retry-After') if exc.headers else None
            if raw_ra is not None:
                retry_after = float(raw_ra)
        except (ValueError, TypeError):
            pass
        raise ProviderHttpError(status_code=int(exc.code), body=error_body, retry_after=retry_after) from exc
    except urllib.error.URLError as exc:
        raise ProviderTimeoutError(str(exc)) from exc


async def _http_post_json_async(
    endpoint: str,
    payload: dict,
    headers: Dict[str, str],
    timeout_s: float,
    *,
    return_headers: bool = False,
) -> dict | tuple[dict, Dict[str, str]]:
    """Async variant — offloads blocking urllib call to a thread pool."""
    import asyncio
    return await asyncio.to_thread(
        _http_post_json, endpoint, payload, headers, timeout_s,
        return_headers=return_headers,
    )


def _with_model_fallback(
    models_to_try: list[str],
    try_model: Callable[[str], str | None],
    debug: bool,
) -> str:
    """Try each model in order; on ProviderHttpError try next; on parse miss return None to retry."""
    last_error: ProviderHttpError | None = None
    for model in models_to_try:
        try:
            result = try_model(model)
            if result is not None:
                return result
        except ProviderHttpError as exc:
            last_error = exc

    if last_error is not None:
        raise _map_http_error(last_error)
    raise ProviderResponseError(message='provider request failed without response')
