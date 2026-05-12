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

from ..retry import NonRetryableError, RetryExhausted, with_retry

DEFAULT_RETRY_HTTP_CODES: Set[int] = {502, 503, 504, 524}


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


def _anthropic_invoke_factory(
    *,
    api_key: str,
    model: str,
    temperature: float,
    timeout_s: float,
    max_retries: int,
    retry_backoff_s: float,
    retry_http_codes: Set[int],
    base_url: str = '',
    enable_native_tools: bool = False,
) -> InvokeFn:
    endpoint = (base_url.rstrip('/') + '/messages') if base_url else 'https://api.anthropic.com/v1/messages'
    headers = {
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
    }

    def _request_once(prompt: str) -> dict:
        request_prompt = prompt
        payload = {
            'model': model,
            'max_tokens': 4096 if enable_native_tools else 1024,
            'temperature': temperature,
            'messages': [{'role': 'user', 'content': request_prompt}],
        }
        if (
            enable_native_tools
            and _prompt_requires_file_tool(prompt)
            and not _prompt_has_successful_tool_result(prompt)
        ):
            request_prompt = _native_tool_prompt(prompt)
            payload['messages'] = [{'role': 'user', 'content': request_prompt}]
            payload['tools'] = _anthropic_file_tools()
            if _prompt_should_force_write_file(prompt):
                payload['tool_choice'] = {'type': 'tool', 'name': 'write_file'}
            else:
                payload['tool_choice'] = {'type': 'any'}
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
        try:
            response = _request_with_retry(
                request_fn=lambda: _request_once(prompt),
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
                retry_http_codes=retry_http_codes,
            )
            return _extract_anthropic_text(response)
        except ProviderHttpError as exc:
            error_msg = f'Anthropic API error: HTTP {exc.status_code}'
            if exc.body:
                error_msg += f' - {exc.body[:200]}'
            raise ValueError(error_msg) from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            detail = str(exc).strip()
            message = 'invalid Anthropic response format'
            if detail:
                message = f'{message}: {detail}'
            raise ValueError(message) from exc

    return invoke


def _extract_text_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return ''.join(_extract_text_value(item) for item in value).strip()
    if not isinstance(value, dict):
        return ''

    for key in (
        'text',
        'output_text',
        'value',
        'answer',
        'final',
        'response',
        'output',
        'thinking',
        'reasoning',
        'reasoning_content',
        'message_content',
    ):
        text = value.get(key)
        if isinstance(text, str):
            return text
        nested_text = _extract_text_value(text)
        if nested_text:
            return nested_text

    for key in ('content', 'message', 'delta'):
        text = _extract_text_value(value.get(key))
        if text:
            return text

    choices = value.get('choices')
    if isinstance(choices, list):
        for choice in choices:
            text = _extract_text_value(choice)
            if text:
                return text

    return ''


def _extract_anthropic_tool_use_json(response: dict) -> str:
    content = response.get('content')
    if not isinstance(content, list):
        return ''

    tool_calls: list[dict[str, object]] = []
    thoughts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get('type')
        if block_type == 'text':
            text = block.get('text')
            if isinstance(text, str) and text.strip():
                thoughts.append(text.strip())
            continue
        if block_type != 'tool_use':
            continue
        name = block.get('name')
        call_id = block.get('id')
        arguments = block.get('input')
        if not isinstance(name, str) or not isinstance(call_id, str):
            continue
        if not isinstance(arguments, dict):
            arguments = {}
        tool_calls.append({'id': call_id, 'name': name, 'arguments': arguments})

    if not tool_calls:
        return ''
    return json.dumps(
        {
            'thought': '\n'.join(thoughts),
            'plan': [],
            'tool_calls': tool_calls,
            'final': None,
        },
        ensure_ascii=False,
    )


def _extract_anthropic_text(response: dict) -> str:
    error = response.get('error')
    if isinstance(error, dict):
        message = error.get('message')
        if isinstance(message, str) and message.strip():
            raise ValueError(message)

    native_tool_json = _extract_anthropic_tool_use_json(response)
    if native_tool_json:
        return native_tool_json

    text = _extract_text_value(response).strip()
    if text:
        return text
    keys = ','.join(sorted(str(key) for key in response.keys()))
    raise ValueError(f'no text content; keys={keys or "(none)"}')


def _anthropic_file_tools() -> list[dict[str, object]]:
    return [
        {
            'name': 'read_file',
            'description': 'Read one UTF-8 file inside the workspace.',
            'input_schema': {
                'type': 'object',
                'properties': {'path': {'type': 'string'}},
                'required': ['path'],
            },
        },
        {
            'name': 'write_file',
            'description': 'Write one UTF-8 file inside the workspace, creating parent directories as needed.',
            'input_schema': {
                'type': 'object',
                'properties': {
                    'path': {'type': 'string'},
                    'content': {'type': 'string'},
                },
                'required': ['path', 'content'],
            },
        },
        {
            'name': 'apply_patch',
            'description': 'Apply a unified patch to files inside the workspace.',
            'input_schema': {
                'type': 'object',
                'properties': {'patch': {'type': 'string'}},
                'required': ['patch'],
            },
        },
        {
            'name': 'search',
            'description': 'Search text in files inside the workspace.',
            'input_schema': {
                'type': 'object',
                'properties': {'pattern': {'type': 'string'}},
                'required': ['pattern'],
            },
        },
        {
            'name': 'run_command',
            'description': 'Run a command in the workspace.',
            'input_schema': {
                'type': 'object',
                'properties': {'cmd': {'type': 'array', 'items': {'type': 'string'}}},
                'required': ['cmd'],
            },
        },
    ]


def _prompt_goal(prompt: str) -> str:
    marker = '\nGoal:\n'
    if marker not in prompt:
        return prompt
    after_goal = prompt.split(marker, 1)[1]
    return after_goal.split('\nHistory:\n', 1)[0]


def _prompt_section(prompt: str, start: str, end: str) -> str:
    if start in prompt:
        value = prompt.split(start, 1)[1]
    elif start.startswith('\n') and prompt.startswith(start[1:]):
        value = prompt[len(start) - 1 :]
    else:
        return ''
    if end in value:
        value = value.split(end, 1)[0]
    return value.strip()


def _native_tool_prompt(prompt: str) -> str:
    goal = _prompt_goal(prompt).strip()
    state_summary = _prompt_section(prompt, '\nStateSummary:\n', '\nLastSteps:\n')
    last_steps = _prompt_section(prompt, '\nLastSteps:\n', '\nToolResults:\n')
    parts = [
        'You are a coding agent. Use the provided tools to perform the user request.',
        'Do not answer with JSON tool_calls text. Call the tool through the API.',
        'For directory plus file creation, call write_file on the target file path; it creates parent directories.',
        'If the user asks for the current path, use StateSummary.workspace.root.',
        '',
        'Goal:',
        goal,
    ]
    if state_summary:
        parts.extend(['', 'StateSummary:', state_summary])
    if last_steps and last_steps != '[]':
        parts.extend(['', 'LastSteps:', last_steps])
    return '\n'.join(parts)


def _prompt_has_successful_tool_result(prompt: str) -> bool:
    tool_results = _prompt_section(prompt, '\nToolResults:\n', '\nOnly output JSON.')
    return (
        "'ok': True" in tool_results
        or '"ok": true' in tool_results
        or '"ok": True' in tool_results
    )


def _prompt_requires_file_tool(prompt: str) -> bool:
    goal = _prompt_goal(prompt).lower()
    action_tokens = (
        '\u65b0\u589e',
        '\u521b\u5efa',
        '\u65b0\u5efa',
        '\u5199\u5165',
        '\u5199\u5230',
        '\u4fee\u6539',
        '\u5220\u9664',
        '\u67e5\u770b',
        '\u68c0\u67e5',
        'create',
        'write',
        'edit',
        'delete',
        'inspect',
        'read',
    )
    target_tokens = (
        '\u6587\u4ef6',
        '\u6587\u4ef6\u5939',
        '\u76ee\u5f55',
        '.txt',
        '.md',
        '.json',
        'file',
        'folder',
        'directory',
    )
    return any(token in goal for token in action_tokens) and any(token in goal for token in target_tokens)


def _prompt_should_force_write_file(prompt: str) -> bool:
    goal = _prompt_goal(prompt).lower()
    write_tokens = (
        '\u65b0\u589e',
        '\u521b\u5efa',
        '\u65b0\u5efa',
        '\u5199\u5165',
        '\u5199\u5230',
        'create',
        'write',
    )
    file_tokens = ('\u6587\u4ef6', '.txt', '.md', '.json', 'file')
    return any(token in goal for token in write_tokens) and any(token in goal for token in file_tokens)


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
            if exc.body:
                error_msg += f' - {exc.body[:200]}'
            raise ValueError(error_msg) from exc
        except (KeyError, IndexError) as exc:
            raise ValueError('invalid Gemini response format') from exc

    return invoke


def anthropic_invoke_factory(
    *,
    api_key: str,
    model: str,
    temperature: float = 0.2,
    timeout_s: float = 60.0,
    base_url: str = '',
    debug: bool = False,
    enable_native_tools: bool = False,
) -> InvokeFn:
    """Public wrapper for Anthropic provider with optional custom base_url."""
    return _anthropic_invoke_factory(
        api_key=api_key, model=model, temperature=temperature,
        timeout_s=timeout_s, max_retries=2, retry_backoff_s=1.0,
        retry_http_codes={502, 503, 504, 524}, base_url=base_url,
        enable_native_tools=enable_native_tools,
    )


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
    )


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
            'User-Agent': 'Anvil/0.1 (+https://github.com/t0ugh-sys/Anvil)',
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
        last_error: Optional[ProviderHttpError] = None

        for current_model in models_to_try:
            try:
                data = _request_with_retry(
                    request_fn=lambda: _request_once(messages, current_model),
                    max_retries=max_retries,
                    retry_backoff_s=retry_backoff_s,
                    retry_http_codes=retry_http_codes,
                )
            except ProviderHttpError as exc:
                last_error = exc
                continue

            choices = data.get('choices', [])
            if not choices:
                continue
            first = choices[0]
            if not isinstance(first, dict):
                continue
            message = first.get('message', {})
            if not isinstance(message, dict):
                continue
            content = message.get('content', '')
            if not isinstance(content, str):
                continue
            return content

        if last_error is not None:
            if debug:
                raise ValueError(f'HTTP {last_error.status_code}: {last_error.body}')
            raise ValueError(
                f'HTTP {last_error.status_code}: request failed (enable --provider-debug for details)'
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
            'User-Agent': 'Anvil/0.1 (+https://github.com/t0ugh-sys/Anvil)',
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
        last_error: Optional[ProviderHttpError] = None

        for current_model in models_to_try:
            try:
                data = _request_with_retry(
                    request_fn=lambda: _request_once(prompt, current_model),
                    max_retries=max_retries,
                    retry_backoff_s=retry_backoff_s,
                    retry_http_codes=retry_http_codes,
                )
            except ProviderHttpError as exc:
                last_error = exc
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

        if last_error is not None:
            if debug:
                raise ValueError(f'HTTP {last_error.status_code}: {last_error.body}')
            raise ValueError(
                f'HTTP {last_error.status_code}: request failed (enable --provider-debug for details)'
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
        base_url = str(getattr(args, 'base_url', '')).strip()
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
            base_url=base_url,
            temperature=temperature,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            retry_http_codes=retry_http_codes,
            enable_native_tools=(mode == 'coding'),
        )
    if provider == 'gemini':
        api_key_env = str(getattr(args, 'api_key_env', 'GEMINI_API_KEY'))
        api_key = os.getenv(api_key_env, '').strip()
        if not api_key:
            raise ValueError(f'api key is missing: env {api_key_env}')
        base_url = str(getattr(args, 'base_url', '')).strip()
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
            base_url=base_url,
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


# Provider registry for programmatic access
_PROVIDER_REGISTRY = {
    'mock': 'Mock provider for testing',
    'openai_compatible': 'OpenAI-compatible API (OpenAI, Ollama, etc.)',
    'anthropic': 'Anthropic Claude API',
    'gemini': 'Google Gemini API',
}


def list_providers() -> dict[str, str]:
    """List all available providers and their descriptions."""
    return _PROVIDER_REGISTRY.copy()


def get_provider(name: str) -> InvokeFn | None:
    """Get a provider invoke function by name.
    
    Returns None if provider requires configuration (api_key, base_url, etc.)
    """
    if name == 'mock':
        return _mock_invoke_factory('mock-model', mode='json')
    # Other providers require configuration, return None
    # Use build_invoke_from_args for full configuration
    return None
