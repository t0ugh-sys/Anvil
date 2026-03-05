from __future__ import annotations

import itertools

from loop_agent.core.agent import LoopAgent
from loop_agent.core.types import StopConfig
from loop_agent.steps.json_loop import JsonLoopState, make_json_decision_step


def main() -> None:
    responses = itertools.cycle(
        [
            '{"answer":"第一版：还不够好","done":false}',
            '{"answer":"第二版：更接近目标","done":false}',
            '{"answer":"最终版：满足目标","done":true}',
        ]
    )

    def invoke(_: str) -> str:
        return next(responses)

    step = make_json_decision_step(invoke, history_window=2)
    agent = LoopAgent(step=step, stop=StopConfig(max_steps=10, max_elapsed_s=10.0))
    result = agent.run(goal='给我一句 20 字以内的中文自我介绍', initial_state=JsonLoopState())

    print('done:', result.done)
    print('stop_reason:', result.stop_reason.value)
    print('steps:', result.steps)
    print('final_output:', result.final_output)


if __name__ == '__main__':
    main()

