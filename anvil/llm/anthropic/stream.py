from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable, Dict, List

from .._http import ProviderHttpError
from ..usage import TokenUsageTracker
from .chat import AnthropicChatResponse


def anthropic_stream_invoke_factory(
    *,
    api_key: str,
    model: str,
    temperature: float = 0.2,
    timeout_s: float = 120.0,
    base_url: str = '',
    debug: bool = False,
    usage_tracker: TokenUsageTracker | None = None,
    thinking_budget_tokens: int = 0,
    enable_prompt_caching: bool = True,
    system_prompt: str = '',
    on_chunk: Callable[[str], None] | None = None,
) -> Callable[[List[Dict[str, object]]], AnthropicChatResponse]:
    """Streaming Anthropic provider for real-time responses.

    Uses SSE (Server-Sent Events) to stream responses, providing
    better UX for long-running requests. Returns the complete
    AnthropicChatResponse after streaming finishes.

    For extended thinking, streaming shows thinking progress in real-time,
    which is especially valuable for complex reasoning tasks.
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

    def _request_once_stream(messages: List[Dict[str, object]]) -> dict:
        payload: Dict[str, object] = {
            'model': model,
            'max_tokens': _build_max_tokens(),
            'temperature': temperature,
            'messages': messages,
            'stream': True,
        }

        if thinking_budget_tokens > 0:
            payload['thinking'] = {
                'type': 'enabled',
                'budget_tokens': thinking_budget_tokens,
            }
            payload['temperature'] = 1.0

        if system_prompt:
            payload['system'] = [
                {
                    'type': 'text',
                    'text': system_prompt,
                    'cache_control': {'type': 'ephemeral'} if enable_prompt_caching else {},
                }
            ]

        body = json.dumps(payload).encode('utf-8')
        request = urllib.request.Request(endpoint, data=body, headers=headers, method='POST')

        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                # Parse SSE stream
                text_parts: List[str] = []
                thinking_blocks: List[Dict[str, str]] = []
                current_thinking = ''
                usage_data: Dict[str, int] = {}

                for line in response:
                    line_str = line.decode('utf-8', errors='replace').strip()
                    if not line_str or line_str.startswith(':'):
                        continue
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        if data_str == '[DONE]':
                            break
                        try:
                            event = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        event_type = event.get('type', '')

                        if event_type == 'content_block_start':
                            block = event.get('content_block', {})
                            if block.get('type') == 'thinking':
                                current_thinking = ''
                        elif event_type == 'content_block_delta':
                            delta = event.get('delta', {})
                            delta_type = delta.get('type', '')
                            if delta_type == 'thinking_delta':
                                current_thinking += delta.get('thinking', '')
                            elif delta_type == 'text_delta':
                                chunk = delta.get('text', '')
                                if chunk:
                                    text_parts.append(chunk)
                                    if on_chunk is not None:
                                        on_chunk(chunk)
                        elif event_type == 'content_block_stop':
                            if current_thinking:
                                thinking_blocks.append({
                                    'type': 'thinking',
                                    'thinking': current_thinking,
                                })
                                current_thinking = ''
                        elif event_type == 'message_delta':
                            usage_delta = event.get('usage', {})
                            for k, v in usage_delta.items():
                                usage_data[k] = usage_data.get(k, 0) + (v if isinstance(v, int) else 0)
                        elif event_type == 'message_start':
                            start_usage = event.get('message', {}).get('usage', {})
                            for k, v in start_usage.items():
                                usage_data[k] = usage_data.get(k, 0) + (v if isinstance(v, int) else 0)

                # Track usage
                if usage_tracker is not None and usage_data:
                    usage_tracker.record(usage_data, model=model)

                return {
                    'text': ''.join(text_parts),
                    'thinking_blocks': thinking_blocks,
                    'usage': usage_data,
                }

        except urllib.error.HTTPError as exc:
            error_body = ''
            try:
                error_body = exc.read().decode('utf-8', errors='replace')
            except Exception:
                error_body = str(exc)
            raise ProviderHttpError(status_code=int(exc.code), body=error_body) from exc

    def invoke(messages: List[Dict[str, object]]) -> AnthropicChatResponse:
        try:
            result = _request_once_stream(messages)
            return AnthropicChatResponse(
                text=result.get('text', ''),
                thinking_blocks=result.get('thinking_blocks', []),
                raw_content=[],
            )
        except ProviderHttpError as exc:
            error_msg = f'Anthropic streaming error: HTTP {exc.status_code}'
            if debug and exc.body:
                error_msg += f' - {exc.body[:200]}'
            raise ValueError(error_msg) from exc

    return invoke
