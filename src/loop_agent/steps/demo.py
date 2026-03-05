from __future__ import annotations

from dataclasses import dataclass

from ..core.types import StepContext, StepResult


@dataclass(frozen=True)
class DemoState:
    attempt: int = 0


def demo_step(context: StepContext[DemoState]) -> StepResult[DemoState]:
    next_attempt = context.state.attempt + 1
    output = f'第 {next_attempt} 次尝试：目标是「{context.goal}」'
    done = next_attempt >= 3
    return StepResult(output=output, state=DemoState(attempt=next_attempt), done=done)

