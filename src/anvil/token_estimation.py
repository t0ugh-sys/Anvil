"""
Token estimation with hybrid approach.

Three-tier strategy (inspired by Claude Code):
1. Rough heuristic: char/4 for text, char/2 for JSON, 2000 for images
2. API usage integration: use actual usage data from last response
3. Message-aware: walk message objects extracting text/image/tool_use content

File-type-aware ratios reduce estimation error from ~30% to ~10%.

CJK characters (Chinese/Japanese/Korean) typically tokenize to ~1.5-2 tokens each,
while ASCII characters average ~0.25 tokens (4 chars per token).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

__all__ = [
    'estimate_tokens',
    'estimate_text_tokens',
    'estimate_content_tokens',
    'estimate_message_tokens',
    'estimate_messages_tokens',
    'TokenUsage',
    'extract_usage',
    'HybridTokenCounter',
]

# Token estimation constants
CHARS_PER_TOKEN_DEFAULT = 4
CHARS_PER_TOKEN_JSON = 2
IMAGE_TOKENS_ESTIMATE = 2000
MESSAGE_OVERHEAD_TOKENS = 10
TOOL_USE_OVERHEAD_TOKENS = 50
ROLE_OVERHEAD_TOKENS = 4

# CJK Unified Ideographs and extensions
_CJK_RANGES = (
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
    (0x2A700, 0x2B73F),  # CJK Unified Ideographs Extension C
    (0x2B740, 0x2B81F),  # CJK Unified Ideographs Extension D
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    (0x3000, 0x303F),    # CJK Symbols and Punctuation
    (0xFF00, 0xFFEF),    # Halfwidth and Fullwidth Forms
    (0x3040, 0x309F),    # Hiragana
    (0x30A0, 0x30FF),    # Katakana
    (0xAC00, 0xD7AF),    # Hangul Syllables
)

# Pre-built range objects for O(1) containment checks
_CJK_RANGE_SET = tuple(range(start, end + 1) for start, end in _CJK_RANGES)


def _is_cjk(char: str) -> bool:
    """Check if a character is CJK (Chinese/Japanese/Korean)."""
    code = ord(char)
    return any(code in r for r in _CJK_RANGE_SET)


def _count_cjk(text: str) -> int:
    """Count CJK characters in text."""
    return sum(1 for ch in text if _is_cjk(ch))


def estimate_text_tokens(text: str, *, is_json: bool = False) -> int:
    """Estimate tokens for a text string with CJK awareness.

    CJK characters: ~1.5 tokens per character.
    ASCII/Latin characters: ~0.25 tokens per character (4 chars per token).
    JSON content uses char/2 ratio (more tokens per char due to syntax).
    """
    if not text:
        return 0

    if is_json:
        # JSON: treat as mostly ASCII with syntax overhead
        return max(1, len(text) // CHARS_PER_TOKEN_JSON)

    # CJK-aware estimation
    cjk_count = _count_cjk(text)
    ascii_count = len(text) - cjk_count
    # CJK: ~1.5 tokens per char, ASCII: ~0.25 tokens per char (4 chars/token)
    return max(1, int(cjk_count * 1.5 + ascii_count * 0.25))


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
    """Estimate tokens for a sequence of text parts (backward compatible, CJK-aware)."""
    total = sum(estimate_text_tokens(part) for part in parts if part)
    return max(1, total) if total else 0


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
        self._last_total_chars: int = 0  # chars from last calibrated call
        self._chars_per_token: float = CHARS_PER_TOKEN_DEFAULT

    def update_from_response(
        self,
        response: Dict[str, Any],
        message_count: int,
        total_chars: int = 0,
    ) -> None:
        """Update calibration from API response usage data."""
        usage = extract_usage(response)
        if usage.input_tokens > 0 and message_count > 0:
            self._last_usage = usage
            self._last_message_count = message_count
            self._last_total_chars = total_chars

    def estimate_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate tokens for messages, using API calibration if available."""
        if self._last_usage is not None and self._last_total_chars > 0:
            total_chars = sum(
                len(json.dumps(msg, ensure_ascii=False)) if isinstance(msg.get('content'), list)
                else len(str(msg.get('content', '')))
                for msg in messages
            )
            if total_chars <= 0:
                return 0
            # Use calibrated chars-per-token ratio from last API response
            chars_per_token = self._last_total_chars / self._last_usage.input_tokens
            return max(1, int(total_chars / chars_per_token))

        # Fallback to heuristic
        return estimate_messages_tokens(messages)

    @property
    def has_api_data(self) -> bool:
        return self._last_usage is not None
