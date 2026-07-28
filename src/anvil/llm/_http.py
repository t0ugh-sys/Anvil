from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable, Dict, Set

from ..retry import NonRetryableError, RetryExhausted, with_retry
from ._types import DEFAULT_RETRY_HTTP_CODES


class ProviderHttpError(Exception):
    """HTTP error from LLM provider API."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f'HTTP {status_code}: {body}')
        self.status_code = status_code
        self.body = body


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
        )
    except (RetryExhausted, NonRetryableError) as exc:
        # Convert to ProviderHttpError for backward compatibility
        status = getattr(exc, 'last_status_code', None) or getattr(exc, 'status_code', 0)
        body = getattr(exc, 'last_body', None) or getattr(exc, 'body', '')
        raise ProviderHttpError(status_code=status, body=body) from exc


def _http_post_json(
    endpoint: str,
    payload: dict,
    headers: Dict[str, str],
    timeout_s: float,
) -> dict:
    """Shared HTTP POST helper — serialises payload, sends request, returns parsed JSON.

    Eliminates 4x duplicated request/error-handling boilerplate across providers.
    """
    body = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(endpoint, data=body, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode('utf-8')
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        error_body = ''
        try:
            error_body = exc.read().decode('utf-8', errors='replace')
        except Exception:
            error_body = str(exc)
        raise ProviderHttpError(status_code=int(exc.code), body=error_body) from exc


def _with_model_fallback(
    models_to_try: list[str],
    try_model: Callable[[str], str | None],
    debug: bool,
) -> str:
    """Try each model in order; on ProviderHttpError try next; on parse miss return None to retry.

    Eliminates duplicated fallback + error-raising boilerplate across providers.
    """
    last_error: ProviderHttpError | None = None
    for model in models_to_try:
        try:
            result = try_model(model)
            if result is not None:
                return result
        except ProviderHttpError as exc:
            last_error = exc

    if last_error is not None:
        if debug:
            raise ValueError(f'HTTP {last_error.status_code}: {last_error.body}')
        raise ValueError(
            f'HTTP {last_error.status_code}: request failed (enable --provider-debug for details)'
        )
    raise ValueError('provider request failed without response')
