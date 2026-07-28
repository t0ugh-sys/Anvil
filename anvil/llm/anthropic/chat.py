from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .._http import ProviderHttpError, _http_post_json, _request_with_retry
from ..usage import TokenUsageTracker


@dataclass
class AnthropicChatResponse:
    """Structured response from Anthropic chat with thinking support.

    Contains both the text response and any thinking blocks,
    enabling multi-turn conversations that preserve thinking context.
    """
    text: str
    thinking_blocks: List[Dict[str, str]] = None  # [{'type': 'thinking', 'thinking': '...'}]
    raw_content: List[Dict] = None  # Full content blocks from API

    def __post_init__(self):
        if self.thinking_blocks is None:
            self.thinking_blocks = []
        if self.raw_content is None:
            self.raw_content = []

    def to_assistant_message(self) -> Dict[str, object]:
        """Convert to assistant message format for multi-turn passback.

        Claude API requires thinking blocks to be passed back in subsequent
        messages for conversation continuity with extended thinking.
        """
        content = []
        for block in self.thinking_blocks:
            content.append({
                'type': 'thinking',
                'thinking': block.get('thinking', ''),
            })
        if self.text:
            content.append({
                'type': 'text',
                'text': self.text,
            })
        return {'role': 'assistant', 'content': content}


def anthropic_chat_invoke_factory(
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
    system_prompt: str = '',
) -> Callable[[List[Dict[str, object]]], AnthropicChatResponse]:
    """Anthropic provider with full messages array support and thinking block passback.

    Unlike `anthropic_invoke_factory` which takes a single prompt string,
    this factory accepts a messages array for proper multi-turn conversations.
    It returns an `AnthropicChatResponse` that includes thinking blocks,
    which must be passed back in subsequent messages per Claude API requirements.

    Usage::

        invoke = anthropic_chat_invoke_factory(
            api_key='...', model='claude-sonnet-4-20250514',
            thinking_budget_tokens=10000,
            system_prompt='You are a coding assistant.',
        )
        # First turn
        response = invoke([{'role': 'user', 'content': 'Solve this...'}])
        # Second turn — pass back thinking blocks
        messages = [
            {'role': 'user', 'content': 'Solve this...'},
            response.to_assistant_message(),
            {'role': 'user', 'content': 'Now explain your reasoning.'},
        ]
        response2 = invoke(messages)
    """
    endpoint = (base_url.rstrip('/') + '/messages') if base_url else 'https://api.anthropic.com/v1/messages'
    headers = {
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
    }

    def _build_max_tokens() -> int:
        if thinking_budget_tokens > 0:
            return max(thinking_budget_tokens + 4096, thinking_budget_tokens * 2)
        return 4096

    def _request_once(messages: List[Dict[str, object]]) -> dict:
        payload: Dict[str, object] = {
            'model': model,
            'max_tokens': _build_max_tokens(),
            'temperature': temperature,
            'messages': messages,
        }

        if stop_sequences:
            payload['stop_sequences'] = stop_sequences

        # Extended thinking
        if thinking_budget_tokens > 0:
            payload['thinking'] = {
                'type': 'enabled',
                'budget_tokens': thinking_budget_tokens,
            }
            payload['temperature'] = 1.0

        # System prompt with cache_control
        if system_prompt:
            payload['system'] = [
                {
                    'type': 'text',
                    'text': system_prompt,
                    'cache_control': {'type': 'ephemeral'} if enable_prompt_caching else {},
                }
            ]

        return _http_post_json(endpoint, payload, headers, timeout_s)

    def invoke(messages: List[Dict[str, object]]) -> AnthropicChatResponse:
        try:
            response = _request_with_retry(
                request_fn=lambda: _request_once(messages),
                max_retries=2,
                retry_backoff_s=1.0,
                retry_http_codes={502, 503, 504, 524},
            )

            # Track usage
            if usage_tracker is not None:
                usage = response.get('usage', {})
                if isinstance(usage, dict):
                    usage_tracker.record(usage, model=model)

            # Parse response content blocks
            content = response.get('content', [])
            text_parts: List[str] = []
            thinking_blocks: List[Dict[str, str]] = []

            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get('type')
                if block_type == 'thinking':
                    thinking_text = block.get('thinking', '')
                    if isinstance(thinking_text, str) and thinking_text.strip():
                        thinking_blocks.append({
                            'type': 'thinking',
                            'thinking': thinking_text,
                        })
                elif block_type == 'text':
                    text = block.get('text', '')
                    if isinstance(text, str) and text.strip():
                        text_parts.append(text.strip())

            return AnthropicChatResponse(
                text='\n'.join(text_parts),
                thinking_blocks=thinking_blocks,
                raw_content=content,
            )

        except ProviderHttpError as exc:
            error_msg = f'Anthropic API error: HTTP {exc.status_code}'
            if debug and exc.body:
                error_msg += f' - {exc.body[:200]}'
            raise ValueError(error_msg) from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            detail = str(exc).strip()
            message = 'invalid Anthropic response format'
            if detail:
                message = f'{message}: {detail}'
            raise ValueError(message) from exc

    return invoke
