from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .agent_protocol import ToolResult, parse_agent_step, render_agent_step_schema
from .compression import (
    CompactManager,
    CompressionConfig,
    TranscriptEntry,
    archive_transcript,
    estimate_tokens,
    micro_compact_entries,
    summarize_entries_deterministically,
)
from .core.types import StepContext, StepResult
from .policies import ToolPolicy
from .task_graph import TaskGraph, TaskStatus
from .task_store import TaskStore
from .todo import TodoItem, TodoManager, TodoSnapshot, render_todo_lines
from .tools import ToolContext, ToolDispatchMap, build_default_tools, execute_tool_call

try:
    from .skills import SkillLoader
except ImportError:  # pragma: no cover
    SkillLoader = None  # type: ignore[assignment]


DeciderFn = Callable[[str, Tuple[str, ...], Tuple[ToolResult, ...], Dict[str, object], Tuple[str, ...]], str]
SummarizerFn = Callable[[str, str, Tuple[TranscriptEntry, ...]], str]


@dataclass(frozen=True)
class ToolUseState:
    history: Tuple[str, ...] = tuple()
    tool_results: Tuple[ToolResult, ...] = tuple()
    todos: Tuple[TodoItem, ...] = tuple()
    rounds_since_todo_update: int = 0
    transcript: Tuple[TranscriptEntry, ...] = tuple()
    compact_summary: str = ''
    compaction_count: int = 0
    archived_transcripts: Tuple[str, ...] = tuple()
    last_compaction_reason: str = ''


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


def _build_task_state_summary(task_store: TaskStore | None) -> Dict[str, object]:
    if task_store is None:
        return {'enabled': False}

    task_files = task_store.list_task_files()
    if not task_files:
        return {
            'enabled': True,
            'root_dir': str(task_store.root_dir),
            'counts': {'total': 0},
            'pending': [],
            'ready': [],
            'running': [],
            'blocked': [],
            'completed': [],
            'failed': [],
        }

    graph = task_store.load_graph()
    return _summarize_task_graph(graph, task_store=task_store)


def _summarize_task_graph(graph: TaskGraph, *, task_store: TaskStore) -> Dict[str, object]:
    tasks = graph.tasks()

    def collect(status: TaskStatus) -> List[Dict[str, str]]:
        return [
            {'id': task.id, 'title': task.title}
            for task in tasks
            if task.status == status
        ]

    return {
        'enabled': True,
        'root_dir': str(task_store.root_dir),
        'counts': {
            'total': len(tasks),
            'pending': sum(1 for task in tasks if task.status == TaskStatus.pending),
            'ready': sum(1 for task in tasks if task.status == TaskStatus.ready),
            'running': sum(1 for task in tasks if task.status == TaskStatus.running),
            'blocked': sum(1 for task in tasks if task.status == TaskStatus.blocked),
            'completed': sum(1 for task in tasks if task.status == TaskStatus.completed),
            'failed': sum(1 for task in tasks if task.status == TaskStatus.failed),
        },
        'pending': collect(TaskStatus.pending),
        'ready': collect(TaskStatus.ready),
        'running': collect(TaskStatus.running),
        'blocked': collect(TaskStatus.blocked),
        'completed': collect(TaskStatus.completed),
        'failed': collect(TaskStatus.failed),
    }


def _augment_state_summary(
    context: StepContext[ToolUseState],
    *,
    nag_after_rounds: int,
    skills: Optional['SkillLoader'] = None,
    task_store: TaskStore | None = None,
    compression_config: CompressionConfig | None = None,
) -> Dict[str, object]:
    summary = dict(context.state_summary)
    summary['todo_state'] = _build_todo_state_summary(context.state, nag_after_rounds=nag_after_rounds)
    summary['task_state'] = _build_task_state_summary(task_store)
    config = compression_config or CompressionConfig()
    summary['compression_state'] = {
        'summary': context.state.compact_summary,
        'compaction_count': context.state.compaction_count,
        'archived_transcripts': list(context.state.archived_transcripts[-5:]),
        'recent_transcript': [
            entry.render_line()
            for entry in context.state.transcript[-config.recent_transcript_entries :]
        ],
        'estimated_tokens': estimate_tokens(
            [context.state.compact_summary, *[entry.content for entry in context.state.transcript]]
        ),
        'last_compaction_reason': context.state.last_compaction_reason,
    }
    reminder = summary['todo_state'].get('reminder')
    if reminder:
        summary['todo_reminder'] = reminder
    if skills is not None:
        summary['available_skills'] = skills.metadata()
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


def _append_transcript_entries(
    state: ToolUseState,
    *,
    thought: str,
    tool_calls,
    tool_results: List[ToolResult],
) -> Tuple[TranscriptEntry, ...]:
    entries = list(state.transcript)
    entries.append(TranscriptEntry(kind='thought', content=thought))
    for call, result in zip(tool_calls, tool_results):
        content = result.output if result.ok else (result.error or result.output or 'tool error')
        entries.append(
            TranscriptEntry(
                kind='tool_result',
                content=content[:4000],
                tool_name=call.name,
                call_id=result.id,
                ok=result.ok,
            )
        )
    return tuple(entries)


def _compact_state_if_needed(
    *,
    goal: str,
    state: ToolUseState,
    transcripts_dir: Path | None,
    summarizer: SummarizerFn | None,
    compression_config: CompressionConfig,
    compact_manager: CompactManager,
) -> ToolUseState:
    compacted_transcript = micro_compact_entries(
        state.transcript,
        keep_last_results=compression_config.micro_keep_last_results,
    )
    next_state = ToolUseState(
        history=state.history,
        tool_results=state.tool_results,
        todos=state.todos,
        rounds_since_todo_update=state.rounds_since_todo_update,
        transcript=compacted_transcript,
        compact_summary=state.compact_summary,
        compaction_count=state.compaction_count,
        archived_transcripts=state.archived_transcripts,
        last_compaction_reason=state.last_compaction_reason,
    )

    estimated_tokens = estimate_tokens(
        [next_state.compact_summary, *[entry.content for entry in next_state.transcript]]
    )
    reason = ''
    if compact_manager.requested:
        reason = compact_manager.reason or 'manual'
    elif estimated_tokens > compression_config.max_context_tokens:
        reason = f'auto:{estimated_tokens}>{compression_config.max_context_tokens}'

    if not reason:
        return next_state

    archived_transcripts = list(next_state.archived_transcripts)
    if transcripts_dir is not None:
        archive_path = archive_transcript(
            transcripts_dir=transcripts_dir,
            compaction_index=next_state.compaction_count + 1,
            reason=reason,
            goal=goal,
            previous_summary=next_state.compact_summary,
            entries=next_state.transcript,
        )
        archived_transcripts.append(str(archive_path))

    summary = (
        summarizer(goal, next_state.compact_summary, next_state.transcript)
        if summarizer is not None
        else summarize_entries_deterministically(
            goal=goal,
            previous_summary=next_state.compact_summary,
            entries=next_state.transcript,
        )
    )
    return ToolUseState(
        history=next_state.history,
        tool_results=next_state.tool_results,
        todos=next_state.todos,
        rounds_since_todo_update=next_state.rounds_since_todo_update,
        transcript=(TranscriptEntry(kind='summary', content=summary),),
        compact_summary=summary,
        compaction_count=next_state.compaction_count + 1,
        archived_transcripts=tuple(archived_transcripts),
        last_compaction_reason=reason,
    )


def execute_tool_use_round(
    *,
    decider: DeciderFn,
    context: StepContext[ToolUseState],
    tool_context: ToolContext,
    dispatch_map: ToolDispatchMap,
    nag_after_rounds: int = 3,
    skills: Optional['SkillLoader'] = None,
    task_store: TaskStore | None = None,
    compression_config: CompressionConfig | None = None,
    transcripts_dir: Path | None = None,
    summarizer: SummarizerFn | None = None,
) -> StepResult[ToolUseState]:
    config = compression_config or CompressionConfig()
    config.validate()
    augmented_state_summary = _augment_state_summary(
        context,
        nag_after_rounds=nag_after_rounds,
        skills=skills,
        task_store=task_store,
        compression_config=config,
    )
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
        skill_loader=skills,
        compact_manager=CompactManager(),
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
    updated_transcript = _append_transcript_entries(
        context.state,
        thought=parsed.thought,
        tool_calls=parsed.tool_calls,
        tool_results=executed,
    )
    todo_snapshot = todo_manager.snapshot(previous_rounds_since_update=context.state.rounds_since_todo_update)
    draft_state = ToolUseState(
        history=updated_history,
        tool_results=tuple(executed),
        todos=todo_snapshot.items,
        rounds_since_todo_update=todo_snapshot.rounds_since_update,
        transcript=updated_transcript,
        compact_summary=context.state.compact_summary,
        compaction_count=context.state.compaction_count,
        archived_transcripts=context.state.archived_transcripts,
        last_compaction_reason=context.state.last_compaction_reason,
    )
    compacted_state = _compact_state_if_needed(
        goal=context.goal,
        state=draft_state,
        transcripts_dir=transcripts_dir,
        summarizer=summarizer,
        compression_config=config,
        compact_manager=tool_context.compact_manager or CompactManager(),
    )
    new_state = compacted_state
    metadata = _build_round_metadata(
        context=context,
        state_summary=augmented_state_summary,
        thought=parsed.thought,
        plan=parsed.plan,
        tool_calls=parsed.tool_calls,
        tool_results=executed,
    )
    metadata['todo_state'] = _build_todo_state_summary(new_state, nag_after_rounds=nag_after_rounds)
    metadata['compression_state'] = {
        'summary': new_state.compact_summary,
        'compaction_count': new_state.compaction_count,
        'archived_transcripts': list(new_state.archived_transcripts[-5:]),
        'recent_transcript': [entry.render_line() for entry in new_state.transcript[-config.recent_transcript_entries :]],
        'last_compaction_reason': new_state.last_compaction_reason,
    }
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
    task_store: TaskStore | None = None,
    compression_config: CompressionConfig | None = None,
    transcripts_dir: Path | None = None,
    summarizer: SummarizerFn | None = None,
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
            skills=skills,
            task_store=task_store,
            compression_config=compression_config,
            transcripts_dir=transcripts_dir,
            summarizer=summarizer,
        )

    return step
