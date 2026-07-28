from __future__ import annotations

from typing import Set

from ._types import InvokeFn
from ._http import ProviderHttpError, _http_post_json, _request_with_retry


def _gemini_invoke_factory(
    *,
    api_key: str,
    model: str,
    temperature: float,
    timeout_s: float,
    max_retries: int,
    retry_backoff_s: float,
    retry_http_codes: Set[int],
    base_url: str = '',
    debug: bool = False,
) -> InvokeFn:
    resolved_base = base_url.rstrip('/') if base_url else 'https://generativelanguage.googleapis.com/v1'
    endpoint = f'{resolved_base}/models/{model}:generateContent?key={api_key}'
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'Anvil/0.1 (+https://github.com/t0ugh-sys/Anvil)',
    }

    def _request_once(prompt: str) -> dict:
        payload = {
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {'temperature': temperature},
        }
        return _http_post_json(endpoint, payload, headers, timeout_s)

    def invoke(prompt: str) -> str:
        try:
            response = _request_with_retry(
                request_fn=lambda: _request_once(prompt),
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
                retry_http_codes=retry_http_codes,
            )
            candidates = response.get('candidates', [])
            if not candidates:
                raise ValueError('invalid Gemini response: no candidates')
            content = candidates[0].get('content', {})
            parts = content.get('parts', [])
            if not parts:
                raise ValueError('invalid Gemini response: no parts')
            return parts[0].get('text', '')
        except ProviderHttpError as exc:
            error_msg = f'Gemini API error: HTTP {exc.status_code}'
            if debug and exc.body:
                error_msg += f' - {exc.body[:200]}'
            elif exc.body:
                error_msg += f' - {exc.body[:100]}'
            raise ValueError(error_msg) from exc
        except (KeyError, IndexError) as exc:
            raise ValueError('invalid Gemini response format') from exc

    return invoke


def gemini_invoke_factory(
    *,
    api_key: str,
    model: str,
    temperature: float = 0.2,
    timeout_s: float = 60.0,
    base_url: str = '',
    debug: bool = False,
) -> InvokeFn:
    """Public wrapper for Gemini provider with optional custom base_url."""
    return _gemini_invoke_factory(
        api_key=api_key, model=model, temperature=temperature,
        timeout_s=timeout_s, max_retries=2, retry_backoff_s=1.0,
        retry_http_codes={502, 503, 504, 524}, base_url=base_url,
        debug=debug,
    )


__all__ = ['_gemini_invoke_factory', 'gemini_invoke_factory']
