from __future__ import annotations

import argparse
import json
import os
import urllib.request
from typing import Callable

InvokeFn = Callable[[str], str]


def _mock_invoke_factory(model: str) -> InvokeFn:
    state = {'count': 0}

    def invoke(_: str) -> str:
        state['count'] += 1
        if state['count'] >= 2:
            return json.dumps({'answer': f'[{model}] final answer', 'done': True}, ensure_ascii=False)
        return json.dumps({'answer': f'[{model}] draft answer', 'done': False}, ensure_ascii=False)

    return invoke


def _openai_compatible_invoke_factory(
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout_s: float,
) -> InvokeFn:
    endpoint = base_url.rstrip('/') + '/chat/completions'

    def invoke(prompt: str) -> str:
        payload = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': temperature,
        }
        body = json.dumps(payload).encode('utf-8')
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        }
        request = urllib.request.Request(endpoint, data=body, headers=headers, method='POST')
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode('utf-8')
        data = json.loads(raw)
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

    return invoke


def build_invoke_from_args(args: argparse.Namespace) -> InvokeFn:
    provider = str(getattr(args, 'provider', 'mock'))
    model = str(getattr(args, 'model', 'mock-model'))

    if provider == 'mock':
        return _mock_invoke_factory(model=model)

    if provider == 'openai_compatible':
        base_url = str(getattr(args, 'base_url', '')).strip()
        if not base_url:
            raise ValueError('base_url is required for openai_compatible provider')
        api_key_env = str(getattr(args, 'api_key_env', 'OPENAI_API_KEY'))
        api_key = os.getenv(api_key_env, '').strip()
        if not api_key:
            raise ValueError(f'api key is missing: env {api_key_env}')
        temperature = float(getattr(args, 'temperature', 0.2))
        timeout_s = float(getattr(args, 'provider_timeout_s', 60.0))
        return _openai_compatible_invoke_factory(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout_s=timeout_s,
        )

    raise ValueError(f'unknown provider: {provider}')

