from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar

StateT = TypeVar('StateT')


class StopReason(str, Enum):
    done = 'done'
    max_steps = 'max_steps'
    timeout = 'timeout'
    cancelled = 'cancelled'
    step_error = 'step_error'


@dataclass(frozen=True)
class StopConfig:
    max_steps: int = 20
    max_elapsed_s: float = 60.0

    def validate(self) -> None:
        if self.max_steps < 1:
            raise ValueError('max_steps must be >= 1')
        if self.max_elapsed_s <= 0:
            raise ValueError('max_elapsed_s must be > 0')


@dataclass(frozen=True)
class StepContext(Generic[StateT]):
    goal: str
    state: StateT
    step_index: int
    started_at_s: float
    now_s: float
    history: Tuple[str, ...]
    state_summary: Dict[str, Any] = field(default_factory=dict)
    last_steps: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def elapsed_s(self) -> float:
        return self.now_s - self.started_at_s


@dataclass(frozen=True)
class StepResult(Generic[StateT]):
    output: str
    state: StateT
    done: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult(Generic[StateT]):
    final_output: str
    state: StateT
    done: bool
    steps: int
    elapsed_s: float
    history: Tuple[str, ...]
    stop_reason: StopReason
    error: Optional[str ] = None


StepFn = Callable[[StepContext[StateT]], StepResult[StateT]]
CancelFn = Callable[[], bool]
ObserverFn = Callable[[str, Dict[str, Any]], None]


@dataclass(frozen=True)
class ContextSnapshot:
    state_summary: Dict[str, Any] = field(default_factory=dict)
    last_steps: Tuple[str, ...] = field(default_factory=tuple)


ContextProviderFn = Callable[[], ContextSnapshot]


def monotonic_s() -> float:
    return time.monotonic()
