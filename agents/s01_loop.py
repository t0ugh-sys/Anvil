from __future__ import annotations

from dataclasses import dataclass

from loop_agent.core.agent import LoopAgent
from loop_agent.core.types import StepContext, StepResult, StopConfig


@dataclass(frozen=True)
class StageState:
    iteration: int = 0


def stage_step(context: StepContext[StageState]) -> StepResult[StageState]:
    next_iteration = context.state.iteration + 1
    done = next_iteration >= 3
    output = f's01 iteration={next_iteration} done={done}'
    return StepResult(output=output, state=StageState(iteration=next_iteration), done=done)


def main() -> None:
    agent = LoopAgent(step=stage_step, stop=StopConfig(max_steps=5, max_elapsed_s=10.0))
    result = agent.run(goal='learn the minimal loop', initial_state=StageState())
    print(f'done={result.done}')
    print(f'steps={result.steps}')
    print(f'stop_reason={result.stop_reason.value}')
    print(f'final_output={result.final_output}')


if __name__ == '__main__':
    main()
