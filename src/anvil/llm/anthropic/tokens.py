from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from .._http import ProviderHttpError


def anthropic_count_tokens(
    *,
    api_key: str,
    model: str,
    messages: List[Dict[str, object]],
    system_prompt: str = '',
    tools: Optional[List[Dict]] = None,
    base_url: str = '',
    timeout_s: float = 30.0,
) -> Dict[str, int]:
    """Count tokens for a request without making an actual API call.

    Uses Anthropic's token counting endpoint for precise token counts,
    which is more accurate than the estimation in token_estimation.py.

    Args:
        api_key: Anthropic API key
        model: Model identifier (e.g. 'claude-sonnet-5')
        messages: Messages array to count tokens for
        system_prompt: Optional system prompt
        tools: Optional tool definitions
        base_url: Optional custom base URL
        timeout_s: Request timeout

    Returns:
        Dict with 'input_tokens' count and optional 'output_tokens' estimate
    """
    endpoint = (base_url.rstrip('/') + '/v1/messages?beta=true') if base_url else 'https://api.anthropic.com/v1/messages?beta=true'
    headers = {
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
    }

    payload: Dict[str, object] = {
        'model': model,
        'messages': messages,
    }

    if system_prompt:
        payload['system'] = [{'type': 'text', 'text': system_prompt}]

    if tools:
        payload['tools'] = tools

    try:
        body = json.dumps(payload).encode('utf-8')
        request = urllib.request.Request(endpoint, data=body, headers=headers, method='POST')
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            result = json.loads(response.read().decode('utf-8'))
            return {
                'input_tokens': int(result.get('input_tokens', 0)),
            }
    except urllib.error.HTTPError as exc:
        error_body = ''
        try:
            error_body = exc.read().decode('utf-8', errors='replace')
        except Exception:
            error_body = str(exc)
        # If counting endpoint is not available, fall back to estimation
        raise ProviderHttpError(status_code=int(exc.code), body=error_body) from exc
    except Exception as exc:
        raise ValueError(f'Token counting failed: {exc}') from exc


def anthropic_count_tokens_or_estimate(
    *,
    api_key: str = '',
    model: str = 'claude-sonnet-5',
    messages: List[Dict[str, object]],
    system_prompt: str = '',
    tools: Optional[List[Dict]] = None,
    base_url: str = '',
) -> int:
    """Count tokens with automatic fallback to estimation.

    Tries the Anthropic counting API first; if unavailable (no API key,
    rate limited, etc.), falls back to the local estimation algorithm.
    """
    if api_key:
        try:
            result = anthropic_count_tokens(
                api_key=api_key,
                model=model,
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
                base_url=base_url,
            )
            return result['input_tokens']
        except (ProviderHttpError, ValueError):
            pass  # Fall through to estimation

    # Fallback: estimate from content length
    from ...token_estimation import estimate_messages_tokens
    return estimate_messages_tokens(messages)
