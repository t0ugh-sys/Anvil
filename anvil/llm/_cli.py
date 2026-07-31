from __future__ import annotations

import argparse
import os
from typing import Dict, List, Set

from ._types import InvokeFn, DEFAULT_RETRY_HTTP_CODES
from .mock import _mock_invoke_factory
from .gemini import _gemini_invoke_factory
from .openai_compat import _openai_compatible_invoke_factory
from .anthropic import _anthropic_invoke_factory
from .rate_limit import RateLimitTracker


def _parse_common_provider_args(args: argparse.Namespace) -> dict:
    """Extract common provider arguments from CLI namespace."""
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


def build_invoke_from_args(
    args: argparse.Namespace,
    *,
    mode: str = 'json_loop',
    usage_tracker=None,
    rate_limit_tracker: RateLimitTracker | None = None,
) -> InvokeFn:
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
            usage_tracker=usage_tracker,
        )

    if provider == 'anthropic':
        api_key = _resolve_api_key(args, 'ANTHROPIC_API_KEY')
        return _anthropic_invoke_factory(
            api_key=api_key, model=model, base_url=common['base_url'],
            temperature=common['temperature'], timeout_s=common['timeout_s'],
            max_retries=common['max_retries'], retry_backoff_s=common['retry_backoff_s'],
            retry_http_codes=common['retry_http_codes'], debug=common['debug'],
            enable_native_tools=(mode == 'coding'),
            usage_tracker=usage_tracker,
            rate_limit_tracker=rate_limit_tracker,
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
    return None


__all__ = [
    '_parse_common_provider_args',
    '_resolve_api_key',
    'parse_provider_headers',
    'build_invoke_from_args',
    'list_providers',
    'get_provider',
]
