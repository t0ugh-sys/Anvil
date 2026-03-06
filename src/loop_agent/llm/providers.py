from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from typing import Callable, Dict, List, Optional, Set

InvokeFn = Callable[[str], str]
ChatInvokeFn = Callable[[List[Dict[str, str]]], str]

DEFAULT_RETRY_HTTP_CODES = {502, 503, 504, 524}


def _anthropic_invoke_factory(
    *,
    api_key: str,
    model: str,
    temperature: float,
    timeout_s: float,
    max_retries: int,
    retry_backoff_s: float,
    retry_http_codes: Set[int],
) -> InvokeFn:
    endpoint = 'https://api.anthropic.com/v1/messages'
    headers = {
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
        'User-Agent': 'LoopAgent/0.1 (+https://github.com/t0ugh-sys/LoopAgent)',
    }

    def _request_once(prompt: str) -> dict:
        payload = {
            'model': model,
            'max_tokens': 1024,
            'temperature': temperature,
            'messages': [{'role': 'user', 'content': prompt}],
        }
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

    def invoke(prompt: str) -> str:
        last_http_error: Optional[ProviderHttpError ] = None
        for attempt in range(max_retries + 1):
            try:
                response = _request_once(prompt)
                return response['content'][0]['text']
            except ProviderHttpError as exc:
                last_http_error = exc
                if exc.status_code in retry_http_codes and attempt < max_retries:
                    time.sleep(retry_backoff_s * (2 ** attempt))
                    continue
                break

        if last_http_error is not None:
            error_msg = f'Anthropic API error: HTTP {last_http_error.status_code}'
            if last_http_error.body:
                error_msg += f' - {last_http_error.body[:200]}'
            raise ValueError(error_msg)
        raise ValueError('Anthropic API request failed - no response received')

    return invoke


def _gemini_invoke_factory(
    *,
    api_key: str,
    model: str,
    temperature: float,
    timeout_s: float,
    max_retries: int,
    retry_backoff_s: float,
    retry_http_codes: Set[int],
) -> InvokeFn:
    base_url = 'https://generativelanguage.googleapis.com/v1'
    endpoint = f'{base_url}/models/{model}:generateContent?key={api_key}'
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'LoopAgent/0.1 (+https://github.com/t0ugh-sys/LoopAgent)',
    }

    def _request_once(prompt: str) -> dict:
        payload = {
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {'temperature': temperature},
        }
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

    def invoke(prompt: str) -> str:
        last_http_error: Optional[ProviderHttpError ] = None
        for attempt in range(max_retries + 1):
            try:
                response = _request_once(prompt)
                candidates = response.get('candidates', [])
                if candidates and len(candidates) > 0:
                    content = candidates[0].get('content', {})
                    parts = content.get('parts', [])
                    if parts and len(parts) > 0:
                        return parts[0].get('text', '')
                raise ValueError('invalid Gemini response: no candidates/content/parts')
            except ProviderHttpError as exc:
                last_http_error = exc
                if exc.status_code in retry_http_codes and attempt < max_retries:
                    time.sleep(retry_backoff_s * (2 ** attempt))
                    continue
                break

        if last_http_error is not None:
            error_msg = f'Gemini API error: HTTP {last_http_error.status_code}'
            if last_http_error.body:
                error_msg += f' - {last_http_error.body[:200]}'
            raise ValueError(error_msg)
        raise ValueError('Gemini API request failed - no response received')

    return invoke


class ProviderHttpError(Exception):
    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f'HTTP {status_code}: {body}')
        self.status_code = status_code
        self.body = body


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
) -> ChatInvokeFn:
    """Return a chat invoke function that accepts OpenAI chat messages.

    `wire_api` is intentionally not supported here yet; TUI uses chat/completions.
    """

    base = base_url.rstrip('/')
    endpoint = base + '/chat/completions'

    models_to_try = [model, *fallback_models]

    def _request_once(messages: List[Dict[str, str]], current_model: str) -> dict:
        payload = {
            'model': current_model,
            'messages': messages,
            'temperature': temperature,
        }
        body = json.dumps(payload).encode('utf-8')
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'LoopAgent/0.1 (+https://github.com/t0ugh-sys/LoopAgent)',
            'Authorization': f'Bearer {api_key}',
        }
        headers.update(extra_headers)
        request = urllib.request.Request(endpoint, data=body, headers=headers, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                raw = response.read().decode('utf-8')
        except urllib.error.HTTPError as exc:
            error_body = ''
            try:
                error_body = exc.read().decode('utf-8', errors='replace')
            except Exception:
                error_body = ''
            raise ProviderHttpError(status_code=int(exc.code), body=error_body) from exc
        return json.loads(raw)

    def invoke(messages: List[Dict[str, str]]) -> str:
        last_http_error: Optional[ProviderHttpError] = None

        for current_model in models_to_try:
            data = None
            for attempt in range(max_retries + 1):
                try:
                    data = _request_once(messages, current_model)
                    break
                except ProviderHttpError as exc:
                    last_http_error = exc
                    should_retry = exc.status_code in retry_http_codes and attempt < max_retries
                    if should_retry:
                        time.sleep(retry_backoff_s * (2 ** attempt))
                        continue
                    break
            if data is None:
                continue

            choices = data.get('choices', [])
            if not isinstance(choices, list) or not choices:
                raise ValueError('invalid openai-compatible response: choices missing')
            first = choices[0]
            if not isinstance(first, dict):
                raise ValueError('invalid openai-compatible response: choice item invalid')
            message = first.get('message', {})
            if not isinstance(message, dict):
                raise ValueError('invalid openai-compatible response: message invalid')
            content = message.get('content', '')
            if not isinstance(content, str):
                raise ValueError('invalid openai-compatible response: content invalid')
            return content

        if last_http_error is not None:
            if debug:
                raise ValueError(f'HTTP {last_http_error.status_code}: {last_http_error.body}')
            raise ValueError(
                f'HTTP {last_http_error.status_code}: request failed (enable --provider-debug for details)'
            )
        raise ValueError('provider request failed without response')

    return invoke


def _mock_invoke_factory(model: str, *, mode: str) -> InvokeFn:
    state = {'count': 0}

    def invoke(_: str) -> str:
        state['count'] += 1
        if mode == 'coding':
            if state['count'] == 1:
                return json.dumps(
                    {
                        'thought': f'[{model}] read README first',
                        'plan': ['read workspace docs', 'produce final response'],
                        'tool_calls': [{'id': 'call_1', 'name': 'read_file', 'arguments': {'path': 'README.md'}}],
                        'final': None,
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    'thought': f'[{model}] enough context',
                    'plan': [],
                    'tool_calls': [],
                    'final': 'done',
                },
                ensure_ascii=False,
            )
        if state['count'] >= 2:
            return json.dumps({'answer': f'[{model}] final answer', 'done': True}, ensure_ascii=False)
        return json.dumps({'answer': f'[{model}] draft answer', 'done': False}, ensure_ascii=False)

    return invoke


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
        body = json.dumps(payload).encode('utf-8')
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'LoopAgent/0.1 (+https://github.com/t0ugh-sys/LoopAgent)',
            'Authorization': f'Bearer {api_key}',
        }
        headers.update(extra_headers)
        request = urllib.request.Request(endpoint, data=body, headers=headers, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                raw = response.read().decode('utf-8')
        except urllib.error.HTTPError as exc:
            error_body = ''
            try:
                error_body = exc.read().decode('utf-8', errors='replace')
            except Exception:
                error_body = ''
            raise ProviderHttpError(status_code=int(exc.code), body=error_body) from exc
        return json.loads(raw)

    def invoke(prompt: str) -> str:
        last_http_error: Optional[ProviderHttpError ] = None

        for current_model in models_to_try:
            data = None
            for attempt in range(max_retries + 1):
                try:
                    data = _request_once(prompt, current_model)
                    break
                except ProviderHttpError as exc:
                    last_http_error = exc
                    should_retry = exc.status_code in retry_http_codes and attempt < max_retries
                    if should_retry:
                        time.sleep(retry_backoff_s * (2 ** attempt))
                        continue
                    break
            if data is None:
                continue

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
                raise ValueError('invalid responses output: no output_text/content')

            choices = data.get('choices', [])
            if not isinstance(choices, list) or not choices:
                raise ValueError('invalid openai-compatible response: choices missing')
            first = choices[0]
            if not isinstance(first, dict):
                raise ValueError('invalid openai-compatible response: choice item invalid')
            message = first.get('message', {})
            if not isinstance(message, dict):
                raise ValueError('invalid openai-compatible response: message invalid')
            content = message.get('content', '')
            if not isinstance(content, str):
                raise ValueError('invalid openai-compatible response: content invalid')
            return content

        if last_http_error is not None:
            if debug:
                raise ValueError(f'HTTP {last_http_error.status_code}: {last_http_error.body}')
            raise ValueError(
                f'HTTP {last_http_error.status_code}: request failed (enable --provider-debug for details)'
            )
        raise ValueError('provider request failed without response')

    return invoke


def build_invoke_from_args(args: argparse.Namespace, *, mode: str = 'json_loop') -> InvokeFn:
    provider = str(getattr(args, 'provider', 'mock'))
    model = str(getattr(args, 'model', 'mock-model'))

    if provider == 'mock':
        return _mock_invoke_factory(model=model, mode=mode)

    if provider == 'openai_compatible':
        base_url = str(getattr(args, 'base_url', '')).strip()
        if not base_url:
            raise ValueError('base_url is required for openai_compatible provider')
        fallback_models = [item.strip() for item in getattr(args, 'fallback_model', []) if str(item).strip()]
        wire_api = str(getattr(args, 'wire_api', 'chat_completions')).strip()
        if wire_api not in {'chat_completions', 'responses'}:
            raise ValueError('wire_api must be one of: chat_completions,responses')
        api_key_env = str(getattr(args, 'api_key_env', 'OPENAI_API_KEY'))
        api_key = os.getenv(api_key_env, '').strip()
        if not api_key:
            raise ValueError(f'api key is missing: env {api_key_env}')
        temperature = float(getattr(args, 'temperature', 0.2))
        timeout_s = float(getattr(args, 'provider_timeout_s', 60.0))
        debug = bool(getattr(args, 'provider_debug', False))
        extra_headers = parse_provider_headers(getattr(args, 'provider_header', []))
        max_retries = int(getattr(args, 'max_retries', 2))
        retry_backoff_s = float(getattr(args, 'retry_backoff_s', 1.0))
        retry_http_codes = set(int(item) for item in getattr(args, 'retry_http_code', []))
        if not retry_http_codes:
            retry_http_codes = set(DEFAULT_RETRY_HTTP_CODES)
        return _openai_compatible_invoke_factory(
            base_url=base_url,
            api_key=api_key,
            model=model,
            fallback_models=fallback_models,
            temperature=temperature,
            timeout_s=timeout_s,
            wire_api=wire_api,
            debug=debug,
            extra_headers=extra_headers,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            retry_http_codes=retry_http_codes,
        )

    if provider == 'anthropic':
        api_key_env = str(getattr(args, 'api_key_env', 'ANTHROPIC_API_KEY'))
        api_key = os.getenv(api_key_env, '').strip()
        if not api_key:
            raise ValueError(f'api key is missing: env {api_key_env}')
        temperature = float(getattr(args, 'temperature', 0.2))
        timeout_s = float(getattr(args, 'provider_timeout_s', 60.0))
        max_retries = int(getattr(args, 'max_retries', 2))
        retry_backoff_s = float(getattr(args, 'retry_backoff_s', 1.0))
        retry_http_codes = set(int(item) for item in getattr(args, 'retry_http_code', []))
        if not retry_http_codes:
            retry_http_codes = set(DEFAULT_RETRY_HTTP_CODES)
        return _anthropic_invoke_factory(
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            retry_http_codes=retry_http_codes,
        )
    if provider == 'gemini':
        api_key_env = str(getattr(args, 'api_key_env', 'GEMINI_API_KEY'))
        api_key = os.getenv(api_key_env, '').strip()
        if not api_key:
            raise ValueError(f'api key is missing: env {api_key_env}')
        temperature = float(getattr(args, 'temperature', 0.2))
        timeout_s = float(getattr(args, 'provider_timeout_s', 60.0))
        max_retries = int(getattr(args, 'max_retries', 2))
        retry_backoff_s = float(getattr(args, 'retry_backoff_s', 1.0))
        retry_http_codes = set(int(item) for item in getattr(args, 'retry_http_code', []))
        if not retry_http_codes:
            retry_http_codes = set(DEFAULT_RETRY_HTTP_CODES)
        return _gemini_invoke_factory(
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            retry_http_codes=retry_http_codes,
        )
    raise ValueError(f'unknown provider: {provider}')


def parse_provider_headers(items: List[str]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for item in items:
        if ':' not in item:
            raise ValueError('provider header must be Key:Value format')
        key, value = item.split(':', 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError('provider header key must not be empty')
        headers[key] = value
    return headers
