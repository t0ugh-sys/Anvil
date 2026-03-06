from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

from ..core.types import StepContext, StepFn, StepResult
from ..protocols.json_decision import parse_json_decision

InvokeFn = Callable[[str], str]


@dataclass(frozen=True)
class JsonLoopState:
    last_answer: str = ''


def build_json_loop_prompt(*, goal: str, history: Tuple[str, ...], history_window: int) -> str:
    recent_history = list(history[-history_window:]) if history_window > 0 else []
    return f"""
你是一个会持续迭代的助手。你的任务是不断改进答案，直到满足用户目标。

用户目标：
{goal}

历史输出（最近 {history_window} 条）：
{recent_history}

要求：
- 只输出 JSON（不要输出其它文字），格式为：{{"answer": "...", "done": true/false}}
- done 只有在你确信“用户目标已满足”时才设为 true
""".strip()


def make_json_decision_step(invoke: InvokeFn, *, history_window: int = 3) -> StepFn[JsonLoopState]:
    if history_window < 0:
        raise ValueError('history_window must be >= 0')

    def step(context: StepContext[JsonLoopState]) -> StepResult[JsonLoopState]:
        if context.last_steps:
            history = context.last_steps
        else:
            history = context.history
        prompt = build_json_loop_prompt(goal=context.goal, history=history, history_window=history_window)
        if context.state_summary:
            prompt = (
                prompt
                + '\n\n状态摘要（必须优先遵守）：\n'
                + str(context.state_summary)
            )
        raw = invoke(prompt)
        decision = parse_json_decision(raw)
        if decision is None:
            return StepResult(output=raw, state=context.state, done=False, metadata={'parse_error': True})
        return StepResult(
            output=decision.answer,
            state=JsonLoopState(last_answer=decision.answer),
            done=decision.done,
            metadata=dict(decision.raw),
        )

    return step
