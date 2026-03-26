from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .agent_protocol import ToolResult, parse_agent_step, render_agent_step_schema
from .core.types import StepContext, StepResult
from .policies import ToolPolicy
from .todo import TodoItem, TodoManager, TodoSnapshot, render_todo_lines
from .tools import ToolContext, ToolDispatchMap, build_default_tools, execute_tool_call

try:
    from .skills import SkillLoader
except ImportError:  # pragma: no cover
    SkillLoader = None  # type: ignore[assignment]


DeciderFn = Callable[[str, Tuple[str, ...], Tuple[ToolResult, ...], Dict[str, object], Tuple[str, ...]], str]


@dataclass(frozen=True)
class ToolUseState:
    history: Tuple[str, ...] = tuple()
    tool_results: Tuple[ToolResult, ...] = tuple()
    todos: Tuple[TodoItem, ...] = tuple()
    rounds_since_todo_update: int = 0


def build_tool_dispatch(
    *,
    skills: Optional['SkillLoader'] = None,
    extra_tools: Optional[ToolDispatchMap] = None,
) -> ToolDispatchMap:
    dispatch_map = build_default_tools()
    if skills is not None:
        dispatch_map.update(skills.get_tools())
    if extra_tools is not None:
        dispatch_map.update(extra_tools)
    return dispatch_map


def _decide_next_step(
    decider: DeciderFn,
    context: StepContext[ToolUseState],
    state_summary: Dict[str, object],
) -> str:
    return decider(
        context.goal,
        context.state.history,
        context.state.tool_results,
        state_summary,
        context.last_steps,
    )


def _build_todo_state_summary(
    state: ToolUseState,
    *,
    nag_after_rounds: int,
) -> Dict[str, object]:
    todo_lines = render_todo_lines(state.todos)
    summary: Dict[str, object] = {
        'items': [item.to_dict() for item in state.todos],
        'lines': todo_lines,
        'rounds_since_update': state.rounds_since_todo_update,
    }
    has_open_items = any(item.status != 'completed' for item in state.todos)
    if has_open_items and state.rounds_since_todo_update >= nag_after_rounds:
        summary['reminder'] = (
            f'todo list has not been updated for {state.rounds_since_todo_update} rounds; '
            'refresh it if progress changed'
        )
    return summary


def _augment_state_summary(
    context: StepContext[ToolUseState],
    *,
    nag_after_rounds: int,
) -> Dict[str, object]:
    summary = dict(context.state_summary)
    summary['todo_state'] = _build_todo_state_summary(context.state, nag_after_rounds=nag_after_rounds)
    reminder = summary['todo_state'].get('reminder')
    if reminder:
        summary['todo_reminder'] = reminder
    return summary


def _dispatch_tool_calls(
    *,
    tool_context: ToolContext,
    dispatch_map: ToolDispatchMap,
    tool_calls,
) -> List[ToolResult]:
    executed: List[ToolResult] = []
    for tool_call in tool_calls:
        executed.append(execute_tool_call(tool_context, tool_call, dispatch_map))
    return executed


def _append_tool_history(
    *,
    history: Tuple[str, ...],
    thought: str,
    tool_results: List[ToolResult],
) -> Tuple[str, ...]:
    updated_history = list(history)
    updated_history.append(f'thought: {thought}')
    for item in tool_results:
        status = 'ok' if item.ok else f'error={item.error}'
        updated_history.append(f'tool[{item.id}] {status}')
    return tuple(updated_history)


def _build_round_metadata(
    *,
    context: StepContext[ToolUseState],
    state_summary: Dict[str, object],
    thought: str,
    plan,
    tool_calls,
    tool_results: List[ToolResult],
) -> Dict[str, object]:
    return {
        'thought': thought,
        'plan': plan,
        'tool_calls': [
            {'id': call.id, 'name': call.name, 'arguments': dict(call.arguments)}
            for call in tool_calls
        ],
        'tool_results': [
            {'id': item.id, 'ok': item.ok, 'error': item.error, 'output': item.output[:2000]}
            for item in tool_results
        ],
        'has_tool_calls': len(tool_calls) > 0,
        'state_summary': state_summary,
        'last_steps': list(context.last_steps),
    }


def execute_tool_use_round(
    *,
    decider: DeciderFn,
    context: StepContext[ToolUseState],
    tool_context: ToolContext,
    dispatch_map: ToolDispatchMap,
    nag_after_rounds: int = 3,
) -> StepResult[ToolUseState]:
    augmented_state_summary = _augment_state_summary(context, nag_after_rounds=nag_after_rounds)
    todo_manager = TodoManager(
        TodoSnapshot(
            items=context.state.todos,
            rounds_since_update=context.state.rounds_since_todo_update,
        )
    )
    tool_context = ToolContext(
        workspace_root=tool_context.workspace_root,
        policy=tool_context.policy,
        todo_manager=todo_manager,
    )
    raw = _decide_next_step(decider, context, augmented_state_summary)
    parsed = parse_agent_step(raw)
    if parsed is None:
        output = 'invalid agent step json. expected schema: ' + render_agent_step_schema()
        return StepResult(
            output=output,
            state=context.state,
            done=False,
            metadata={'parse_error': True, 'raw_response': raw[:2000]},
        )

    executed = _dispatch_tool_calls(
        tool_context=tool_context,
        dispatch_map=dispatch_map,
        tool_calls=parsed.tool_calls,
    )
    updated_history = _append_tool_history(
        history=context.state.history,
        thought=parsed.thought,
        tool_results=executed,
    )
    todo_snapshot = todo_manager.snapshot(previous_rounds_since_update=context.state.rounds_since_todo_update)
    new_state = ToolUseState(
        history=updated_history,
        tool_results=tuple(executed),
        todos=todo_snapshot.items,
        rounds_since_todo_update=todo_snapshot.rounds_since_update,
    )
    metadata = _build_round_metadata(
        context=context,
        state_summary=augmented_state_summary,
        thought=parsed.thought,
        plan=parsed.plan,
        tool_calls=parsed.tool_calls,
        tool_results=executed,
    )
    metadata['todo_state'] = _build_todo_state_summary(new_state, nag_after_rounds=nag_after_rounds)
    if not metadata['has_tool_calls']:
        final = parsed.final or parsed.thought or 'done'
        return StepResult(output=final, state=new_state, done=True, metadata=metadata)
    return StepResult(output='continue', state=new_state, done=False, metadata=metadata)


def make_tool_use_step(
    *,
    decider: DeciderFn,
    workspace_root: Path,
    skills: Optional['SkillLoader'] = None,
    policy: ToolPolicy = ToolPolicy.allow_all(),
    extra_tools: Optional[ToolDispatchMap] = None,
    todo_nag_after_rounds: int = 3,
) -> Callable[[StepContext[ToolUseState]], StepResult[ToolUseState]]:
    dispatch_map = build_tool_dispatch(skills=skills, extra_tools=extra_tools)
    tool_context = ToolContext(workspace_root=workspace_root, policy=policy)

    def step(context: StepContext[ToolUseState]) -> StepResult[ToolUseState]:
        return execute_tool_use_round(
            decider=decider,
            context=context,
            tool_context=tool_context,
            dispatch_map=dispatch_map,
            nag_after_rounds=todo_nag_after_rounds,
        )

    return step
