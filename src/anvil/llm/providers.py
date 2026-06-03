from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

InvokeFn = Callable[[str], str]
ChatInvokeFn = Callable[[List[Dict[str, str]]], str]

from ..retry import NonRetryableError, RetryExhausted, with_retry

__all__ = [
    'ProviderHttpError',
    'TokenUsageRecord',
    'TokenUsageTracker',
    'build_invoke_from_args',
    'anthropic_invoke_factory',
    'gemini_invoke_factory',
    'openai_compatible_chat_invoke_factory',
    'list_providers',
    'get_provider',
    'parse_provider_headers',
    'InvokeFn',
    'ChatInvokeFn',
    'PromptCache',
]

DEFAULT_RETRY_HTTP_CODES: Set[int] = {502, 503, 504, 524}


# ============== Token Usage Tracking ==============
# Based on Claude API docs: response.usage contains precise token counts.
# Tracks input_tokens, output_tokens, cache_creation, cache_read.


@dataclass
class TokenUsageRecord:
    """A single API call's token usage."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    model: str = ''

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cache_hit_ratio(self) -> float:
        """Fraction of input tokens served from cache (0.0-1.0)."""
        total = self.input_tokens + self.cache_read_input_tokens
        if total <= 0:
            return 0.0
        return self.cache_read_input_tokens / total


class TokenUsageTracker:
    """Tracks cumulative token usage across API calls.

    Each Anthropic API response includes a ``usage`` field with exact
    token counts. This tracker accumulates them for cost monitoring.

    Usage::

        tracker = TokenUsageTracker()
        invoke = anthropic_invoke_factory(..., usage_tracker=invoke)
        result = invoke(prompt)
        print(tracker.summary())
    """

    def __init__(self) -> None:
        self._records: list[TokenUsageRecord] = []

    def record(self, usage: dict, model: str = '') -> None:
        """Record token usage from an API response."""
        self._records.append(TokenUsageRecord(
            input_tokens=int(usage.get('input_tokens', 0) or 0),
            output_tokens=int(usage.get('output_tokens', 0) or 0),
            cache_creation_input_tokens=int(usage.get('cache_creation_input_tokens', 0) or 0),
            cache_read_input_tokens=int(usage.get('cache_read_input_tokens', 0) or 0),
            model=model,
        ))

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self._records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self._records)

    @property
    def total_cache_creation_tokens(self) -> int:
        return sum(r.cache_creation_input_tokens for r in self._records)

    @property
    def total_cache_read_tokens(self) -> int:
        return sum(r.cache_read_input_tokens for r in self._records)

    @property
    def call_count(self) -> int:
        return len(self._records)

    def summary(self) -> dict:
        return {
            'calls': self.call_count,
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens,
            'cache_creation_tokens': self.total_cache_creation_tokens,
            'cache_read_tokens': self.total_cache_read_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
        }

    def last(self) -> TokenUsageRecord | None:
        return self._records[-1] if self._records else None

    def reset(self) -> None:
        self._records.clear()


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
    debug: bool = False,
    enable_native_tools: bool = False,
    usage_tracker: TokenUsageTracker | None = None,
    stop_sequences: List[str] | None = None,
    thinking_budget_tokens: int = 0,
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
        # Stop sequences — halt generation when encountered
        if stop_sequences:
            payload['stop_sequences'] = stop_sequences
        # Extended thinking — enable Claude's internal reasoning
        if thinking_budget_tokens > 0:
            payload['thinking'] = {
                'type': 'enabled',
                'budget_tokens': thinking_budget_tokens,
            }
            # Extended thinking requires temperature=1
            payload['temperature'] = 1.0
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
        return _http_post_json(endpoint, payload, headers, timeout_s)

    def invoke(prompt: str) -> str:
        try:
            response = _request_with_retry(
                request_fn=lambda: _request_once(prompt),
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
                retry_http_codes=retry_http_codes,
            )
            # Track token usage from response
            if usage_tracker is not None:
                usage = response.get('usage', {})
                if isinstance(usage, dict):
                    usage_tracker.record(usage, model=model)
            return _extract_anthropic_text(response)
        except ProviderHttpError as exc:
            error_msg = f'Anthropic API error: HTTP {exc.status_code}'
            if debug and exc.body:
                error_msg += f' - {exc.body[:200]}'
            elif exc.body:
                error_msg += f' - {exc.body[:100]}'
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
    thinking_content: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get('type')
        if block_type == 'thinking':
            # Extended thinking block — capture reasoning
            thinking_text = block.get('thinking', '')
            if isinstance(thinking_text, str) and thinking_text.strip():
                thinking_content.append(thinking_text.strip())
            continue
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
    # Combine thinking and regular thoughts
    all_thoughts = thinking_content + thoughts
    return json.dumps(
        {
            'thought': '\n'.join(all_thoughts),
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
    """Anthropic native tool definitions with prompt caching support.

    Each tool definition includes cache_control for prompt caching.
    Tool definitions are static across requests, so they are ideal
    cache targets. First request creates the cache (10x cost),
    subsequent requests read from cache (0.1x cost) for 5 minutes.
    """
    tools = [
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
    # Mark the last tool with cache_control for prompt caching.
    # Claude caches everything up to and including the marked block.
    if tools:
        tools[-1]['cache_control'] = {'type': 'ephemeral'}
    return tools


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


def anthropic_invoke_factory(
    *,
    api_key: str,
    model: str,
    temperature: float = 0.2,
    timeout_s: float = 60.0,
    base_url: str = '',
    debug: bool = False,
    enable_native_tools: bool = False,
    usage_tracker: TokenUsageTracker | None = None,
    stop_sequences: List[str] | None = None,
    thinking_budget_tokens: int = 0,
) -> InvokeFn:
    """Public wrapper for Anthropic provider with optional custom base_url."""
    return _anthropic_invoke_factory(
        api_key=api_key, model=model, temperature=temperature,
        timeout_s=timeout_s, max_retries=2, retry_backoff_s=1.0,
        retry_http_codes={502, 503, 504, 524}, base_url=base_url,
        debug=debug, enable_native_tools=enable_native_tools,
        usage_tracker=usage_tracker,
        stop_sequences=stop_sequences,
        thinking_budget_tokens=thinking_budget_tokens,
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
        debug=debug,
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
    usage_tracker: TokenUsageTracker | None = None,
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
            # Track token usage from OpenAI-compatible response
            if usage_tracker is not None:
                usage = data.get('usage', {})
                if isinstance(usage, dict):
                    usage_tracker.record(usage, model=current_model)
            return content

        return _with_model_fallback(models_to_try, try_model, debug)

    return invoke


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


def _parse_common_provider_args(args: argparse.Namespace) -> dict:
    """Extract common provider arguments from CLI namespace.

    Eliminates 3x duplicated arg-parsing boilerplate across providers.
    """
    temperature = float(getattr(args, 'temperature', 0.2))
    timeout_s = float(getattr(args, 'provider_timeout_s', 60.0))
    debug = bool(getattr(args, 'provider_debug', False))
    max_retries = int(getattr(args, 'max_retries', 2))
    retry_backoff_s = float(getattr(args, 'retry_backoff_s', 1.0))
    retry_http_codes = set(int(item) for item in getattr(args, 'retry_http_code', []))
    if not retry_http_codes:
        retry_http_codes = set(DEFAULT_RETRY_HTTP_CODES)
    base_url = str(getattr(args, 'base_url', '')).strip()
    return {
        'temperature': temperature,
        'timeout_s': timeout_s,
        'debug': debug,
        'max_retries': max_retries,
        'retry_backoff_s': retry_backoff_s,
        'retry_http_codes': retry_http_codes,
        'base_url': base_url,
    }


def _resolve_api_key(args: argparse.Namespace, default_env: str) -> str:
    """Resolve API key from args and environment."""
    api_key_env = str(getattr(args, 'api_key_env', default_env))
    api_key = os.getenv(api_key_env, '').strip()
    if not api_key:
        raise ValueError(f'api key is missing: env {api_key_env}')
    return api_key


def build_invoke_from_args(args: argparse.Namespace, *, mode: str = 'json_loop') -> InvokeFn:
    provider = str(getattr(args, 'provider', 'mock'))
    model = str(getattr(args, 'model', 'mock-model'))

    if provider == 'mock':
        return _mock_invoke_factory(model=model, mode=mode)

    common = _parse_common_provider_args(args)

    if provider == 'openai_compatible':
        if not common['base_url']:
            raise ValueError('base_url is required for openai_compatible provider')
        fallback_models = [item.strip() for item in getattr(args, 'fallback_model', []) if str(item).strip()]
        wire_api = str(getattr(args, 'wire_api', 'chat_completions')).strip()
        if wire_api not in {'chat_completions', 'responses'}:
            raise ValueError('wire_api must be one of: chat_completions,responses')
        api_key = _resolve_api_key(args, 'OPENAI_API_KEY')
        extra_headers = parse_provider_headers(getattr(args, 'provider_header', []))
        return _openai_compatible_invoke_factory(
            base_url=common['base_url'], api_key=api_key, model=model,
            fallback_models=fallback_models, temperature=common['temperature'],
            timeout_s=common['timeout_s'], wire_api=wire_api, debug=common['debug'],
            extra_headers=extra_headers, max_retries=common['max_retries'],
            retry_backoff_s=common['retry_backoff_s'], retry_http_codes=common['retry_http_codes'],
        )

    if provider == 'anthropic':
        api_key = _resolve_api_key(args, 'ANTHROPIC_API_KEY')
        return _anthropic_invoke_factory(
            api_key=api_key, model=model, base_url=common['base_url'],
            temperature=common['temperature'], timeout_s=common['timeout_s'],
            max_retries=common['max_retries'], retry_backoff_s=common['retry_backoff_s'],
            retry_http_codes=common['retry_http_codes'], debug=common['debug'],
            enable_native_tools=(mode == 'coding'),
        )

    if provider == 'gemini':
        api_key = _resolve_api_key(args, 'GEMINI_API_KEY')
        return _gemini_invoke_factory(
            api_key=api_key, model=model, base_url=common['base_url'],
            temperature=common['temperature'], timeout_s=common['timeout_s'],
            max_retries=common['max_retries'], retry_backoff_s=common['retry_backoff_s'],
            retry_http_codes=common['retry_http_codes'], debug=common['debug'],
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


# ============== Prompt Cache ==============
# Based on Zero2Agent article #42: Cost Optimization & Token Management
# Hash-based cache for identical prompts to avoid redundant API calls.


import hashlib
import threading


class PromptCache:
    """LRU-style cache for LLM prompt responses.

    Caches responses keyed by (model, prompt_hash) to avoid redundant
    API calls for identical prompts. Useful for repeated system prompts,
    template rendering, and test/development scenarios.

    Usage::

        cache = PromptCache(max_size=100)
        key = cache.make_key('claude-3', 'Summarize this: ...')
        cached = cache.get(key)
        if cached is None:
            result = call_llm(prompt)
            cache.set(key, result)
        else:
            result = cached
    """

    def __init__(self, max_size: int = 128, ttl_seconds: float = 3600.0) -> None:
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._cache: Dict[str, tuple] = {}  # key -> (result, timestamp)
        self._access_order: list[str] = []  # LRU tracking
        self._lock = threading.Lock()

    @staticmethod
    def make_key(model: str, prompt: str) -> str:
        """Create a cache key from model name and prompt text."""
        content = f'{model}:{prompt}'
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

    def get(self, key: str) -> str | None:
        """Get cached result if available and not expired."""
        with self._lock:
            if key not in self._cache:
                return None

            result, timestamp = self._cache[key]
            # Check TTL
            import time as _t
            if _t.time() - timestamp > self._ttl_seconds:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return None

            # Move to end (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return result

    def set(self, key: str, result: str) -> None:
        """Store a result in the cache."""
        import time as _t
        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self._max_size and self._access_order:
                oldest = self._access_order.pop(0)
                self._cache.pop(oldest, None)

            self._cache[key] = (result, _t.time())
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()

    @property
    def size(self) -> int:
        return len(self._cache)

    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            'size': len(self._cache),
            'max_size': self._max_size,
            'ttl_seconds': self._ttl_seconds,
        }
