"""Backward-compatibility shim — re-exports everything from the llm subpackage."""
from __future__ import annotations

from ._types import InvokeFn, ChatInvokeFn, DEFAULT_RETRY_HTTP_CODES
from ._http import ProviderHttpError, _http_post_json, _request_with_retry, _with_model_fallback
from .usage import TokenUsageRecord, TokenUsageTracker, CostTracker
from .cache import PromptCache
from .mock import _mock_invoke_factory
from .gemini import _gemini_invoke_factory, gemini_invoke_factory
from .openai_compat import _openai_compatible_invoke_factory, openai_compatible_chat_invoke_factory
from ._cli import (
    _parse_common_provider_args,
    _resolve_api_key,
    parse_provider_headers,
    build_invoke_from_args,
    list_providers,
    get_provider,
)
from .anthropic import (
    _extract_text_value,
    _extract_anthropic_text,
    _extract_anthropic_tool_use_json,
    _anthropic_file_tools,
    _prompt_goal,
    _prompt_section,
    _native_tool_prompt,
    _prompt_has_successful_tool_result,
    _prompt_requires_file_tool,
    _prompt_should_force_write_file,
    _split_system_user,
    _anthropic_invoke_factory,
    anthropic_invoke_factory,
    AnthropicChatResponse,
    anthropic_chat_invoke_factory,
    anthropic_stream_invoke_factory,
    anthropic_count_tokens,
    anthropic_count_tokens_or_estimate,
    BatchRequest,
    BatchResult,
    AnthropicBatchClient,
)

__all__ = [
    'InvokeFn',
    'ChatInvokeFn',
    'DEFAULT_RETRY_HTTP_CODES',
    'ProviderHttpError',
    'TokenUsageRecord',
    'TokenUsageTracker',
    'CostTracker',
    'PromptCache',
    '_mock_invoke_factory',
    '_gemini_invoke_factory',
    'gemini_invoke_factory',
    '_openai_compatible_invoke_factory',
    'openai_compatible_chat_invoke_factory',
    'parse_provider_headers',
    'build_invoke_from_args',
    'list_providers',
    'get_provider',
    '_anthropic_invoke_factory',
    'anthropic_invoke_factory',
    'AnthropicChatResponse',
    'anthropic_chat_invoke_factory',
    'anthropic_stream_invoke_factory',
    'anthropic_count_tokens',
    'anthropic_count_tokens_or_estimate',
    'BatchRequest',
    'BatchResult',
    'AnthropicBatchClient',
]
