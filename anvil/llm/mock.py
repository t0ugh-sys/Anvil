from __future__ import annotations

import json

from ._types import InvokeFn


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
                    'final': f'[mock:{model}] Task complete.',
                },
                ensure_ascii=False,
            )
        if state['count'] >= 2:
            return json.dumps({'answer': f'[{model}] final answer', 'done': True}, ensure_ascii=False)
        return json.dumps({'answer': f'[{model}] draft answer', 'done': False}, ensure_ascii=False)

    return invoke


__all__ = ['_mock_invoke_factory']
