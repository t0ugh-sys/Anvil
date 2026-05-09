"""
Token estimation with hybrid approach.

Three-tier strategy (inspired by Claude Code):
1. Rough heuristic: char/4 for text, char/2 for JSON, 2000 for images
2. API usage integration: use actual usage data from last response
3. Message-aware: walk message objects extracting text/image/tool_use content

File-type-aware ratios reduce estimation error from ~30% to ~10%.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

# Token estimation constants
CHARS_PER_TOKEN_DEFAULT = 4
CHARS_PER_TOKEN_JSON = 2
IMAGE_TOKENS_ESTIMATE = 2000
MESSAGE_OVERHEAD_TOKENS = 10
TOOL_USE_OVERHEAD_TOKENS = 50
ROLE_OVERHEAD_TOKENS = 4


def estimate_text_tokens(text: str, *, is_json: bool = False) -> int:
    """Estimate tokens for a text string.

    JSON content uses char/2 ratio (more tokens per char due to syntax).
    Default text uses char/4 ratio.
    """
    if not text:
        return 0
    chars_per_token = CHARS_PER_TOKEN_JSON if is_json else CHARS_PER_TOKEN_DEFAULT
    return max(1, len(text) // chars_per_token)


def _is_json_like(text: str) -> bool:
    """Check if text looks like JSON content."""
    stripped = text.strip()
    if not stripped:
        return False
    return (stripped[0] in '{[' and stripped[-1] in ']}') or stripped.startswith('```json')


def estimate_content_tokens(content: Any) -> int:
    """Estimate tokens for message content (string or content blocks)."""
    if isinstance(content, str):
        return estimate_text_tokens(content, is_json=_is_json_like(content))

    if isinstance(content, list):
        total = 0
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get('type', '')

            if block_type == 'text':
                text = block.get('text', '')
                total += estimate_text_tokens(text, is_json=_is_json_like(text))

            elif block_type == 'image':
                total += IMAGE_TOKENS_ESTIMATE

            elif block_type == 'tool_use':
                # Tool use blocks have name + JSON arguments
                name = block.get('name', '')
                total += estimate_text_tokens(name)
                args = block.get('input') or block.get('arguments', {})
                if isinstance(args, dict):
                    total += estimate_text_tokens(json.dumps(args), is_json=True)
                total += TOOL_USE_OVERHEAD_TOKENS

            elif block_type == 'tool_result':
                result_content = block.get('content', '')
                total += estimate_content_tokens(result_content)
                total += TOOL_USE_OVERHEAD_TOKENS

            elif block_type == 'thinking':
                text = block.get('thinking', '')
                total += estimate_text_tokens(text)

            else:
                # Unknown block type, estimate from stringified form
                total += estimate_text_tokens(str(block))

        return total

    if content is None:
        return 0

    return estimate_text_tokens(str(content))


def estimate_message_tokens(message: Dict[str, Any]) -> int:
    """Estimate tokens for a single message."""
    total = ROLE_OVERHEAD_TOKENS

    role = message.get('role', '')
    total += estimate_text_tokens(role)

    content = message.get('content')
    if content is not None:
        total += estimate_content_tokens(content)

    # Tool calls in assistant messages
    tool_calls = message.get('tool_calls', [])
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if isinstance(tc, dict):
                fn = tc.get('function', {})
                if isinstance(fn, dict):
                    total += estimate_text_tokens(fn.get('name', ''))
                    args = fn.get('arguments', '')
                    if isinstance(args, str):
                        total += estimate_text_tokens(args, is_json=_is_json_like(args))
                total += TOOL_USE_OVERHEAD_TOKENS

    return total + MESSAGE_OVERHEAD_TOKENS


def estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate total tokens for a list of messages."""
    if not messages:
        return 0
    return sum(estimate_message_tokens(msg) for msg in messages)


def estimate_tokens(parts: Sequence[str]) -> int:
    """Estimate tokens for a sequence of text parts (backward compatible)."""
    total_chars = sum(len(part) for part in parts if part)
    return max(1, total_chars // CHARS_PER_TOKEN_DEFAULT) if total_chars else 0


# ============== API Usage Integration ==============

@dataclass
class TokenUsage:
    """Token usage from API response."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def extract_usage(response: Dict[str, Any]) -> TokenUsage:
    """Extract token usage from API response."""
    usage = response.get('usage', {})
    if not isinstance(usage, dict):
        return TokenUsage()
    return TokenUsage(
        input_tokens=int(usage.get('input_tokens', 0) or 0),
        output_tokens=int(usage.get('output_tokens', 0) or 0),
        cache_creation_input_tokens=int(usage.get('cache_creation_input_tokens', 0) or 0),
        cache_read_input_tokens=int(usage.get('cache_read_input_tokens', 0) or 0),
    )


class HybridTokenCounter:
    """Hybrid token counter using heuristic + API usage data.

    Starts with rough heuristic, switches to actual API usage
    once available. Provides the best estimate at minimal cost.
    """

    def __init__(self) -> None:
        self._last_usage: TokenUsage | None = None
        self._last_message_count: int = 0
        self._chars_per_token: float = CHARS_PER_TOKEN_DEFAULT

    def update_from_response(self, response: Dict[str, Any], message_count: int) -> None:
        """Update calibration from API response usage data."""
        usage = extract_usage(response)
        if usage.input_tokens > 0 and message_count > 0:
            self._last_usage = usage
            self._last_message_count = message_count

    def estimate_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate tokens for messages, using API calibration if available."""
        if self._last_usage is not None and self._last_message_count > 0:
            # Use calibrated chars_per_token from last API response
            total_chars = sum(
                len(json.dumps(msg, ensure_ascii=False)) if isinstance(msg.get('content'), list)
                else len(str(msg.get('content', '')))
                for msg in messages
            )
            # Compute chars per token from last actual usage
            calibrated_cpt = total_chars / self._last_usage.input_tokens if self._last_usage.input_tokens > 0 else CHARS_PER_TOKEN_DEFAULT
            return max(1, int(total_chars / calibrated_cpt))

        # Fallback to heuristic
        return estimate_messages_tokens(messages)

    @property
    def has_api_data(self) -> bool:
        return self._last_usage is not None
