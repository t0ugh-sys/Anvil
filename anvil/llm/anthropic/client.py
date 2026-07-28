from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional, Set, Tuple

from .._types import InvokeFn
from .._http import ProviderHttpError, _http_post_json, _request_with_retry
from ..usage import TokenUsageTracker
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
    stop_sequences: List[str] | None = None,
    thinking_budget_tokens: int = 0,
    enable_prompt_caching: bool = True,
) -> InvokeFn:
    endpoint = (base_url.rstrip('/') + '/messages') if base_url else 'https://api.anthropic.com/v1/messages'
    headers = {
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
    }

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
    """Public wrapper for Anthropic provider with optional custom base_url.

    Args:
        enable_prompt_caching: When True (default), split system prompt from
            user prompt and send with cache_control for 90% cost savings on
            repeated calls. Requires ≥1024 tokens in the cached portion.
    """
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
