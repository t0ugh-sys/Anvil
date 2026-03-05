from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from .types import CancelFn, RunResult, StepContext, StepFn, StepResult, StopConfig, StopReason, monotonic_s

StateT = TypeVar('StateT')


@dataclass(frozen=True)
class LoopAgent(Generic[StateT]):
    step: StepFn[StateT]
    stop: StopConfig = StopConfig()

    def run(self, *, goal: str, initial_state: StateT, is_cancelled: CancelFn | None = None) -> RunResult[StateT]:
        self.stop.validate()
        if not goal.strip():
            raise ValueError('goal must not be empty')

        started_at_s = monotonic_s()
        state = initial_state
        history: list[str] = []
        last_output = ''

        for step_index in range(self.stop.max_steps):
            if is_cancelled is not None and is_cancelled():
                elapsed_s = monotonic_s() - started_at_s
                return RunResult(
                    final_output=last_output,
                    state=state,
                    done=False,
                    steps=step_index,
                    elapsed_s=elapsed_s,
                    history=tuple(history),
                    stop_reason=StopReason.cancelled,
                )

            now_s = monotonic_s()
            elapsed_s = now_s - started_at_s
            if elapsed_s >= self.stop.max_elapsed_s:
                return RunResult(
                    final_output=last_output,
                    state=state,
                    done=False,
                    steps=step_index,
                    elapsed_s=elapsed_s,
                    history=tuple(history),
                    stop_reason=StopReason.timeout,
                )

            context = StepContext(
                goal=goal,
                state=state,
                step_index=step_index,
                started_at_s=started_at_s,
                now_s=now_s,
                history=tuple(history),
            )
            try:
                result: StepResult[StateT] = self.step(context)
            except Exception as exc:
                error_elapsed_s = monotonic_s() - started_at_s
                return RunResult(
                    final_output=last_output,
                    state=state,
                    done=False,
                    steps=step_index,
                    elapsed_s=error_elapsed_s,
                    history=tuple(history),
                    stop_reason=StopReason.step_error,
                    error=str(exc),
                )

            last_output = result.output
            state = result.state
            history.append(result.output)

            if result.done:
                done_elapsed_s = monotonic_s() - started_at_s
                return RunResult(
                    final_output=last_output,
                    state=state,
                    done=True,
                    steps=step_index + 1,
                    elapsed_s=done_elapsed_s,
                    history=tuple(history),
                    stop_reason=StopReason.done,
                )

        exhausted_elapsed_s = monotonic_s() - started_at_s
        return RunResult(
            final_output=last_output,
            state=state,
            done=False,
            steps=self.stop.max_steps,
            elapsed_s=exhausted_elapsed_s,
            history=tuple(history),
            stop_reason=StopReason.max_steps,
        )
