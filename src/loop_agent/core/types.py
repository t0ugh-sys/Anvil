from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Generic, TypeVar

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
    history: tuple[str, ...]

    @property
    def elapsed_s(self) -> float:
        return self.now_s - self.started_at_s


@dataclass(frozen=True)
class StepResult(Generic[StateT]):
    output: str
    state: StateT
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult(Generic[StateT]):
    final_output: str
    state: StateT
    done: bool
    steps: int
    elapsed_s: float
    history: tuple[str, ...]
    stop_reason: StopReason
    error: str | None = None


StepFn = Callable[[StepContext[StateT]], StepResult[StateT]]
CancelFn = Callable[[], bool]


def monotonic_s() -> float:
    return time.monotonic()
