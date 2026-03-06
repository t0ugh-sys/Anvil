from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generic, List, Optional, TypeVar

from .types import (
    CancelFn,
    ContextProviderFn,
    ContextSnapshot,
    ObserverFn,
    RunResult,
    StepContext,
    StepFn,
    StepResult,
    StopConfig,
    StopReason,
    monotonic_s,
)

StateT = TypeVar('StateT')


@dataclass(frozen=True)
class LoopAgent(Generic[StateT]):
    step: StepFn[StateT]
    stop: StopConfig = StopConfig()

    def run(
        self,
        *,
        goal: str,
        initial_state: StateT,
        is_cancelled: Optional[CancelFn ] = None,
        observer: Optional[ObserverFn ] = None,
        context_provider: Optional[ContextProviderFn ] = None,
    ) -> RunResult[StateT]:
        self.stop.validate()
        if not goal.strip():
            raise ValueError('goal must not be empty')

        def emit(event: str, payload: Dict[str, object]) -> None:
            if observer is None:
                return
            try:
                observer(event, payload)
            except Exception:
                return

        started_at_s = monotonic_s()
        state = initial_state
        history: List[str] = []
        last_output = ''

        for step_index in range(self.stop.max_steps):
            if is_cancelled is not None and is_cancelled():
                elapsed_s = monotonic_s() - started_at_s
                run_result = RunResult(
                    final_output=last_output,
                    state=state,
                    done=False,
                    steps=step_index,
                    elapsed_s=elapsed_s,
                    history=tuple(history),
                    stop_reason=StopReason.cancelled,
                )
                emit('stopped', {'reason': run_result.stop_reason.value, 'step': run_result.steps})
                return run_result

            now_s = monotonic_s()
            elapsed_s = now_s - started_at_s
            if elapsed_s >= self.stop.max_elapsed_s:
                run_result = RunResult(
                    final_output=last_output,
                    state=state,
                    done=False,
                    steps=step_index,
                    elapsed_s=elapsed_s,
                    history=tuple(history),
                    stop_reason=StopReason.timeout,
                )
                emit('stopped', {'reason': run_result.stop_reason.value, 'step': run_result.steps})
                return run_result

            snapshot = context_provider() if context_provider else ContextSnapshot()
            context = StepContext(
                goal=goal,
                state=state,
                step_index=step_index,
                started_at_s=started_at_s,
                now_s=now_s,
                history=tuple(history),
                state_summary=snapshot.state_summary,
                last_steps=snapshot.last_steps,
            )
            emit(
                'step_started',
                {
                    'step': step_index,
                    'elapsed_s': elapsed_s,
                    'state_summary': snapshot.state_summary,
                    'last_steps': list(snapshot.last_steps),
                },
            )
            try:
                result: StepResult[StateT] = self.step(context)
            except Exception as exc:
                error_elapsed_s = monotonic_s() - started_at_s
                run_result = RunResult(
                    final_output=last_output,
                    state=state,
                    done=False,
                    steps=step_index,
                    elapsed_s=error_elapsed_s,
                    history=tuple(history),
                    stop_reason=StopReason.step_error,
                    error=str(exc),
                )
                emit('step_failed', {'step': step_index, 'error': str(exc)})
                emit('stopped', {'reason': run_result.stop_reason.value, 'step': run_result.steps, 'error': run_result.error})
                return run_result

            last_output = result.output
            state = result.state
            history.append(result.output)
            emit(
                'step_succeeded',
                {
                    'step': step_index,
                    'done': result.done,
                    'output': result.output,
                    'metadata': result.metadata,
                },
            )

            if result.done:
                done_elapsed_s = monotonic_s() - started_at_s
                run_result = RunResult(
                    final_output=last_output,
                    state=state,
                    done=True,
                    steps=step_index + 1,
                    elapsed_s=done_elapsed_s,
                    history=tuple(history),
                    stop_reason=StopReason.done,
                )
                emit('stopped', {'reason': run_result.stop_reason.value, 'step': run_result.steps})
                return run_result

        exhausted_elapsed_s = monotonic_s() - started_at_s
        run_result = RunResult(
            final_output=last_output,
            state=state,
            done=False,
            steps=self.stop.max_steps,
            elapsed_s=exhausted_elapsed_s,
            history=tuple(history),
            stop_reason=StopReason.max_steps,
        )
        emit('stopped', {'reason': run_result.stop_reason.value, 'step': run_result.steps})
        return run_result
