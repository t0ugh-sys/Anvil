from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .agent_protocol import ToolResult, parse_agent_step, render_agent_step_schema
from .core.agent import LoopAgent
from .core.types import ContextProviderFn, ObserverFn, RunResult, StepContext, StepResult, StopConfig
from .tools import ToolContext, build_default_tools, execute_tool_call

# Try to import skills module
try:
    from .skills import SkillLoader, build_skill_tools, load_skills_from_args
    HAS_SKILLS = True
except ImportError:
    HAS_SKILLS = False

DeciderFn = Callable[[str, tuple[str, ...], tuple[ToolResult, ...], dict[str, object], tuple[str, ...]], str]


@dataclass(frozen=True)
class CodingAgentState:
    history: tuple[str, ...] = tuple()
    tool_results: tuple[ToolResult, ...] = tuple()


def build_coding_step(
    decider: DeciderFn,
    workspace_root: Path,
    skills: SkillLoader | None = None,
) -> Callable[[StepContext[CodingAgentState]], StepResult[CodingAgentState]]:
    # Start with default tools
    tools = build_default_tools()
    
    # Merge with skill tools if available
    if skills is not None:
        skill_tools = skills.get_tools()
        tools.update(skill_tools)
    
    tool_context = ToolContext(workspace_root=workspace_root)

    def step(context: StepContext[CodingAgentState]) -> StepResult[CodingAgentState]:
        raw = decider(
            context.goal,
            context.state.history,
            context.state.tool_results,
            context.state_summary,
            context.last_steps,
        )
        parsed = parse_agent_step(raw)
        if parsed is None:
            output = (
                'invalid agent step json. expected schema: '
                + render_agent_step_schema()
            )
            return StepResult(
                output=output,
                state=context.state,
                done=False,
                metadata={'parse_error': True, 'raw_response': raw[:2000]},
            )

        executed: list[ToolResult] = []
        for tool_call in parsed.tool_calls:
            executed.append(execute_tool_call(tool_context, tool_call, tools))

        history = list(context.state.history)
        history.append(f'thought: {parsed.thought}')
        for item in executed:
            status = 'ok' if item.ok else f'error={item.error}'
            history.append(f'tool[{item.id}] {status}')

        new_state = CodingAgentState(history=tuple(history), tool_results=tuple(executed))
        metadata = {
            'thought': parsed.thought,
            'plan': parsed.plan,
            'tool_calls': [
                {'id': call.id, 'name': call.name, 'arguments': dict(call.arguments)}
                for call in parsed.tool_calls
            ],
            'tool_results': [
                {'id': item.id, 'ok': item.ok, 'error': item.error, 'output': item.output[:2000]}
                for item in executed
            ],
            'state_summary': context.state_summary,
            'last_steps': list(context.last_steps),
        }
        if parsed.done:
            final = parsed.final or ''
            return StepResult(output=final, state=new_state, done=True, metadata=metadata)
        return StepResult(output='continue', state=new_state, done=False, metadata=metadata)

    return step


def run_coding_agent(
    *,
    goal: str,
    decider: DeciderFn,
    workspace_root: Path,
    stop: StopConfig | None = None,
    observer: ObserverFn | None = None,
    context_provider: ContextProviderFn | None = None,
    skills: SkillLoader | None = None,
) -> RunResult[CodingAgentState]:
    step = build_coding_step(decider, workspace_root=workspace_root, skills=skills)
    agent = LoopAgent(step=step, stop=stop or StopConfig(max_steps=20, max_elapsed_s=60.0))
    return agent.run(
        goal=goal,
        initial_state=CodingAgentState(),
        observer=observer,
        context_provider=context_provider,
    )
