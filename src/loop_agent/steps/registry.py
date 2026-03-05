from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable

from ..core.types import StepFn
from ..llm.providers import build_invoke_from_args
from .demo import DemoState, demo_step
from .json_loop import JsonLoopState, make_json_decision_step

StepBundle = tuple[StepFn[Any], Any]
StepBuilder = Callable[[argparse.Namespace], StepBundle]


@dataclass
class StepRegistry:
    _builders: dict[str, StepBuilder]

    def register(self, name: str, builder: StepBuilder) -> None:
        if not name.strip():
            raise ValueError('step name must not be empty')
        self._builders[name] = builder

    def create(self, name: str, args: argparse.Namespace) -> StepBundle:
        builder = self._builders.get(name)
        if builder is None:
            raise ValueError(f'unknown strategy: {name}')
        return builder(args)

    def names(self) -> list[str]:
        return sorted(self._builders.keys())


def _build_demo_step(_: argparse.Namespace) -> StepBundle:
    return demo_step, DemoState()


def _build_json_stub_step(args: argparse.Namespace) -> StepBundle:
    responses = [
        '{"answer":"第一版：还不够好","done":false}',
        '{"answer":"第二版：更接近目标","done":false}',
        '{"answer":"最终版：满足目标","done":true}',
    ]
    index = {'value': 0}

    def invoke(_: str) -> str:
        response = responses[index['value']]
        if index['value'] < len(responses) - 1:
            index['value'] += 1
        return response

    history_window = int(getattr(args, 'history_window', 3))
    step = make_json_decision_step(invoke, history_window=history_window)
    return step, JsonLoopState()


def _build_json_llm_step(args: argparse.Namespace) -> StepBundle:
    invoke = build_invoke_from_args(args)
    history_window = int(getattr(args, 'history_window', 3))
    step = make_json_decision_step(invoke, history_window=history_window)
    return step, JsonLoopState()


def build_default_registry() -> StepRegistry:
    registry = StepRegistry(_builders={})
    registry.register('demo', _build_demo_step)
    registry.register('json_llm', _build_json_llm_step)
    registry.register('json_stub', _build_json_stub_step)
    return registry
