from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional, Set, Tuple

from .._types import AsyncInvokeFn, InvokeFn
from .._http import ProviderHttpError, _http_post_json, _http_post_json_async, _map_http_error, _request_with_retry
from ...errors import ProviderError, ProviderResponseError
from ..usage import TokenUsageTracker
from ..rate_limit import RateLimitTracker
from ._helpers import (
    _anthropic_file_tools,
    _prompt_requires_file_tool,
    _prompt_has_successful_tool_result,
    _native_tool_prompt,
    _prompt_should_force_write_file,
    _split_system_user,
    _extract_anthropic_text,
)


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
    rate_limit_tracker: RateLimitTracker | None = None,
    stop_sequences: List[str] | None = None,
    thinking_budget_tokens: int = 0,
    enable_prompt_caching: bool = True,
    extra_headers: Dict[str, str] | None = None,
) -> InvokeFn:
    endpoint = (base_url.rstrip('/') + '/messages') if base_url else 'https://api.anthropic.com/v1/messages'
    headers = {
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
        'user-agent': 'anthropic-sdk-python/0.1',
    }
    if extra_headers:
        headers.update(extra_headers)

    def _build_max_tokens() -> int:
        """Calculate max_tokens respecting thinking budget constraints.

        Claude API requires max_tokens > budget_tokens when extended thinking
        is enabled. Default max_tokens should also be reasonable for the task.
        """
        if thinking_budget_tokens > 0:
            # Ensure max_tokens > budget_tokens (API requirement)
            # Use budget + 4096 for output, or at least 2x budget
            return max(thinking_budget_tokens + 4096, thinking_budget_tokens * 2)
        return 4096 if enable_native_tools else 1024

    def _request_once(prompt: str) -> dict:
        request_prompt = prompt
        max_tokens = _build_max_tokens()

        payload: Dict[str, object] = {
            'model': model,
            'max_tokens': max_tokens,
            'temperature': temperature,
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

        # Prompt caching: split system prompt for separate caching
        if enable_prompt_caching and not enable_native_tools:
            system_prompt, user_prompt = _split_system_user(prompt)
            if system_prompt:
                payload['system'] = [
                    {
                        'type': 'text',
                        'text': system_prompt,
                        'cache_control': {'type': 'ephemeral'},
                    }
                ]
                payload['messages'] = [{'role': 'user', 'content': user_prompt}]
            else:
                payload['messages'] = [{'role': 'user', 'content': request_prompt}]
        else:
            payload['messages'] = [{'role': 'user', 'content': request_prompt}]

        # Native tool use (coding mode)
        if (
            enable_native_tools
            and _prompt_requires_file_tool(prompt)
            and not _prompt_has_successful_tool_result(prompt)
        ):
            request_prompt = _native_tool_prompt(prompt)
            if enable_prompt_caching:
                # Split native tool prompt too
                system_prompt, user_prompt = _split_system_user(request_prompt)
                if system_prompt:
                    payload['system'] = [
                        {
                            'type': 'text',
                            'text': system_prompt,
                            'cache_control': {'type': 'ephemeral'},
                        }
                    ]
                    payload['messages'] = [{'role': 'user', 'content': user_prompt}]
                else:
                    payload['messages'] = [{'role': 'user', 'content': request_prompt}]
            else:
                payload['messages'] = [{'role': 'user', 'content': request_prompt}]
            payload['tools'] = _anthropic_file_tools()
            if _prompt_should_force_write_file(prompt):
                payload['tool_choice'] = {'type': 'tool', 'name': 'write_file'}
            else:
                payload['tool_choice'] = {'type': 'any'}

        return _http_post_json(
            endpoint, payload, headers, timeout_s,
            return_headers=(rate_limit_tracker is not None),
        )

    def invoke(prompt: str) -> str:
        try:
            raw = _request_with_retry(
                request_fn=lambda: _request_once(prompt),
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
                retry_http_codes=retry_http_codes,
            )
            if rate_limit_tracker is not None and isinstance(raw, tuple):
                response, resp_headers = raw
                rate_limit_tracker.record_headers(resp_headers)
            else:
                response = raw
            # Track token usage from response
            if usage_tracker is not None:
                usage = response.get('usage', {})
                if isinstance(usage, dict):
                    usage_tracker.record(usage, model=model)
            return _extract_anthropic_text(response)
        except ProviderError:
            raise
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            detail = str(exc).strip()
            message = 'invalid Anthropic response format'
            if detail:
                message = f'{message}: {detail}'
            raise ProviderResponseError(message) from exc

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
    enable_prompt_caching: bool = True,
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
        enable_prompt_caching=enable_prompt_caching,
    )


def _anthropic_async_invoke_factory(
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
    usage_tracker: TokenUsageTracker | None = None,
    stop_sequences: List[str] | None = None,
    thinking_budget_tokens: int = 0,
    enable_prompt_caching: bool = True,
    extra_headers: Dict[str, str] | None = None,
) -> AsyncInvokeFn:
    endpoint = (base_url.rstrip('/') + '/messages') if base_url else 'https://api.anthropic.com/v1/messages'
    headers = {
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
        'user-agent': 'anthropic-sdk-python/0.1',
    }
    if extra_headers:
        headers.update(extra_headers)

    def _build_payload(prompt: str) -> dict:
        max_tokens = (
            max(thinking_budget_tokens + 4096, thinking_budget_tokens * 2)
            if thinking_budget_tokens > 0 else 1024
        )
        payload: Dict[str, object] = {
            'model': model,
            'max_tokens': max_tokens,
            'temperature': 1.0 if thinking_budget_tokens > 0 else temperature,
        }
        if stop_sequences:
            payload['stop_sequences'] = stop_sequences
        if thinking_budget_tokens > 0:
            payload['thinking'] = {'type': 'enabled', 'budget_tokens': thinking_budget_tokens}
        if enable_prompt_caching:
            system_prompt, user_prompt = _split_system_user(prompt)
            if system_prompt:
                payload['system'] = [{'type': 'text', 'text': system_prompt, 'cache_control': {'type': 'ephemeral'}}]
                payload['messages'] = [{'role': 'user', 'content': user_prompt}]
                return payload
        payload['messages'] = [{'role': 'user', 'content': prompt}]
        return payload

    async def invoke(prompt: str) -> str:
        import asyncio
        payload = _build_payload(prompt)
        last_exc: ProviderHttpError | None = None
        backoff = retry_backoff_s
        for attempt in range(max_retries + 1):
            try:
                response = await _http_post_json_async(endpoint, payload, headers, timeout_s)
                if usage_tracker is not None:
                    usage = response.get('usage', {})
                    if isinstance(usage, dict):
                        usage_tracker.record(usage, model=model)
                return _extract_anthropic_text(response)
            except ProviderHttpError as exc:
                if exc.status_code not in retry_http_codes or attempt == max_retries:
                    last_exc = exc
                    break
                await asyncio.sleep(backoff * (2 ** attempt))
        assert last_exc is not None
        raise _map_http_error(last_exc) from last_exc

    return invoke


def anthropic_async_invoke_factory(
    *,
    api_key: str,
    model: str,
    temperature: float = 0.2,
    timeout_s: float = 60.0,
    base_url: str = '',
    debug: bool = False,
    usage_tracker: TokenUsageTracker | None = None,
    stop_sequences: List[str] | None = None,
    thinking_budget_tokens: int = 0,
    enable_prompt_caching: bool = True,
) -> AsyncInvokeFn:
    """Async variant of anthropic_invoke_factory — returns an async invoke closure."""
    return _anthropic_async_invoke_factory(
        api_key=api_key, model=model, temperature=temperature,
        timeout_s=timeout_s, max_retries=2, retry_backoff_s=1.0,
        retry_http_codes={502, 503, 504, 524}, base_url=base_url,
        debug=debug, usage_tracker=usage_tracker,
        stop_sequences=stop_sequences,
        thinking_budget_tokens=thinking_budget_tokens,
        enable_prompt_caching=enable_prompt_caching,
    )
