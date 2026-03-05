from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .agent_protocol import AgentStep, ToolResult, parse_agent_step, render_agent_step_schema
from .core.agent import LoopAgent
from .core.types import RunResult, StepContext, StepResult, StopConfig
from .tools import ToolContext, build_default_tools, execute_tool_call

DeciderFn = Callable[[str, tuple[str, ...], tuple[ToolResult, ...]], str]


@dataclass(frozen=True)
class CodingAgentState:
    history: tuple[str, ...] = tuple()
    tool_results: tuple[ToolResult, ...] = tuple()


def build_coding_step(decider: DeciderFn, workspace_root: Path) -> Callable[[StepContext[CodingAgentState]], StepResult[CodingAgentState]]:
    tools = build_default_tools()
    tool_context = ToolContext(workspace_root=workspace_root)

    def step(context: StepContext[CodingAgentState]) -> StepResult[CodingAgentState]:
        raw = decider(context.goal, context.state.history, context.state.tool_results)
        parsed = parse_agent_step(raw)
        if parsed is None:
            output = (
                'invalid agent step json. expected schema: '
                + render_agent_step_schema()
            )
            return StepResult(output=output, state=context.state, done=False, metadata={'parse_error': True})

        executed: list[ToolResult] = []
        for tool_call in parsed.tool_calls:
            executed.append(execute_tool_call(tool_context, tool_call, tools))

        history = list(context.state.history)
        history.append(f'thought: {parsed.thought}')
        for item in executed:
            status = 'ok' if item.ok else f'error={item.error}'
            history.append(f'tool[{item.id}] {status}')

        new_state = CodingAgentState(history=tuple(history), tool_results=tuple(executed))
        if parsed.done:
            final = parsed.final or ''
            return StepResult(output=final, state=new_state, done=True)
        return StepResult(output='continue', state=new_state, done=False)

    return step


def run_coding_agent(
    *,
    goal: str,
    decider: DeciderFn,
    workspace_root: Path,
    stop: StopConfig | None = None,
) -> RunResult[CodingAgentState]:
    step = build_coding_step(decider, workspace_root=workspace_root)
    agent = LoopAgent(step=step, stop=stop or StopConfig(max_steps=20, max_elapsed_s=60.0))
    return agent.run(goal=goal, initial_state=CodingAgentState())

