from __future__ import annotations

from typing import Dict, List, Set

from ._types import InvokeFn, ChatInvokeFn
from ._http import ProviderHttpError, _http_post_json, _request_with_retry, _with_model_fallback
from .usage import TokenUsageTracker


def _openai_compatible_invoke_factory(
    *,
    base_url: str,
    api_key: str,
    model: str,
    fallback_models: List[str],
    temperature: float,
    timeout_s: float,
    wire_api: str,
    debug: bool,
    extra_headers: Dict[str, str],
    max_retries: int,
    retry_backoff_s: float,
    retry_http_codes: Set[int],
) -> InvokeFn:

    base = base_url.rstrip('/')
    if wire_api == 'responses':
        endpoint = base + '/responses'
    else:
        endpoint = base + '/chat/completions'

    models_to_try = [model, *fallback_models]

    def _request_once(prompt: str, current_model: str) -> dict:
        if wire_api == 'responses':
            payload = {'model': current_model, 'input': prompt, 'temperature': temperature}
        else:
            payload = {
                'model': current_model,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': temperature,
            }
        req_headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'Anvil/0.1 (+https://github.com/t0ugh-sys/Anvil)',
            'Authorization': f'Bearer {api_key}',
        }
        req_headers.update(extra_headers)
        return _http_post_json(endpoint, payload, req_headers, timeout_s)

    def invoke(prompt: str) -> str:
        def try_model(current_model: str) -> str | None:
            data = _request_with_retry(
                request_fn=lambda: _request_once(prompt, current_model),
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
                retry_http_codes=retry_http_codes,
            )
            if wire_api == 'responses':
                output_text = data.get('output_text')
                if isinstance(output_text, str) and output_text:
                    return output_text
                output = data.get('output', [])
                if isinstance(output, list):
                    fragments: List[str] = []
                    for item in output:
                        if not isinstance(item, dict):
                            continue
                        content = item.get('content', [])
                        if not isinstance(content, list):
                            continue
                        for piece in content:
                            if not isinstance(piece, dict):
                                continue
                            text = piece.get('text')
                            if isinstance(text, str):
                                fragments.append(text)
                    merged = ''.join(fragments).strip()
                    if merged:
                        return merged
                return None

            choices = data.get('choices', [])
            if not isinstance(choices, list) or not choices:
                return None
            first = choices[0]
            if not isinstance(first, dict):
                return None
            message = first.get('message', {})
            if not isinstance(message, dict):
                return None
            content = message.get('content', '')
            if not isinstance(content, str):
                return None
            return content

        return _with_model_fallback(models_to_try, try_model, debug)

    return invoke


def openai_compatible_chat_invoke_factory(
    *,
    base_url: str,
    api_key: str,
    model: str,
    fallback_models: List[str],
    temperature: float,
    timeout_s: float,
    debug: bool,
    extra_headers: Dict[str, str],
    max_retries: int,
    retry_backoff_s: float,
    retry_http_codes: Set[int],
    usage_tracker: TokenUsageTracker | None = None,
) -> ChatInvokeFn:
    """Return a chat invoke function that accepts OpenAI chat messages."""

    base = base_url.rstrip('/')
    endpoint = base + '/chat/completions'

    models_to_try = [model, *fallback_models]

    def _request_once(messages: List[Dict[str, str]], current_model: str) -> dict:
        payload = {
            'model': current_model,
            'messages': messages,
            'temperature': temperature,
        }
        req_headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'Anvil/0.1 (+https://github.com/t0ugh-sys/Anvil)',
            'Authorization': f'Bearer {api_key}',
        }
        req_headers.update(extra_headers)
        return _http_post_json(endpoint, payload, req_headers, timeout_s)

    def invoke(messages: List[Dict[str, str]]) -> str:
        def try_model(current_model: str) -> str | None:
            data = _request_with_retry(
                request_fn=lambda: _request_once(messages, current_model),
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
                retry_http_codes=retry_http_codes,
            )
            choices = data.get('choices', [])
            if not choices:
                return None
            first = choices[0]
            if not isinstance(first, dict):
                return None
            message = first.get('message', {})
            if not isinstance(message, dict):
                return None
            content = message.get('content', '')
            if not isinstance(content, str):
                return None
            if usage_tracker is not None:
                usage = data.get('usage', {})
                if isinstance(usage, dict):
                    usage_tracker.record(usage, model=current_model)
            return content

        return _with_model_fallback(models_to_try, try_model, debug)

    return invoke


__all__ = ['_openai_compatible_invoke_factory', 'openai_compatible_chat_invoke_factory']
