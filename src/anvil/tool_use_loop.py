from __future__ import annotations

import re
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .agent_protocol import ToolResult, parse_agent_step, render_agent_step_schema
from .background import BackgroundCommandRunner
from .compression import (
    CompactManager,
    CompactConfig,
    TranscriptEntry,
    archive_transcript,
    estimate_tokens,
    micro_compact_entries,
    summarize_entries_deterministically,
    time_based_micro_compact,
)
from .core.types import StepContext, StepResult
from .policies import ToolPolicy, LoopDetector
from .hooks import HookEvent, HookManager, HookInput, HookResult, build_hook_input_for_tool
from .task_graph import TaskGraph, TaskStatus
from .task_store import TaskStore
from .todo import TodoItem, TodoManager, TodoSnapshot, render_todo_lines
from .tools import ToolContext, ToolDispatchMap, build_default_tools, execute_tool_call

try:
    from .skills import SkillLoader
except ImportError:  # pragma: no cover
    SkillLoader = None  # type: ignore[assignment]

__all__ = [
    'ToolUseState',
    'DeciderFn',
    'SummarizerFn',
    'build_tool_dispatch',
    'execute_tool_use_round',
    'make_tool_use_step',
    '_build_reflection',
]


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
    background_notifications: Tuple[ToolResult, ...] = tuple()

    def replace(self, **kwargs: object) -> 'ToolUseState':
        """Create a new state with only the specified fields changed.

        Avoids repeating all 10 fields when only 1-2 need updating.
        """
        return ToolUseState(**{
            f.name: kwargs.get(f.name, getattr(self, f.name))
            for f in self.__dataclass_fields__.values()  # type: ignore[attr-defined]
        })


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


def _looks_like_file_action(goal: str) -> bool:
    normalized = goal.lower()
    action_tokens = (
        '\u65b0\u589e',
        '\u521b\u5efa',
        '\u65b0\u5efa',
        '\u5199\u5165',
        '\u5199\u5230',
        '\u4fee\u6539',
        '\u5220\u9664',
        'create',
        'write',
        'edit',
        'delete',
    )
    target_tokens = (
        '\u6587\u4ef6',
        '\u6587\u4ef6\u5939',
        '\u76ee\u5f55',
        '.txt',
        '.md',
        '.json',
        'file',
        'folder',
        'directory',
    )
    return any(token in normalized for token in action_tokens) and any(
        token in normalized for token in target_tokens
    )


def _has_successful_file_mutation(state: ToolUseState) -> bool:
    mutating_tools = {'write_file', 'apply_patch', 'run_command'}
    return any(
        entry.kind == 'tool_result'
        and entry.ok is True
        and entry.tool_name in mutating_tools
        for entry in state.transcript
    )


def _build_reflection(tool_results: List[ToolResult]) -> str:
    """Build a Reflexion-style self-critique after tool failures.

    Returns a short prompt injected into state_summary so the decider
    can learn from errors without re-reading raw tool output.
    """
    errors = [r for r in tool_results if not r.ok]
    if not errors:
        return ''
    lines = ['REFLECTION: The last round had tool failures.']
    for e in errors:
        lines.append(f'- [{e.id}] {e.error or "unknown error"}')
    lines.append('Consider: What went wrong? What should you try differently?')
    return '\n'.join(lines)


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


def _extract_key_constraints(goal: str) -> List[str]:
    """Extract key constraints from goal text to prevent context drift.

    Uses simple heuristics to identify constraints, deadlines, version
    requirements, and other important parameters that should be
    force-injected into every decider call.
    """
    constraints: List[str] = []
    patterns = [
        (r'(?:must|should|always|never|do not|don\'t)\s+.{10,80}', 'rule'),
        (r'(?:Python|Node|Java|Go|Rust)\s+\d+\.\d+', 'version'),
        (r'(?:deadline|by|before|until)\s+\w+\s+\d+', 'deadline'),
        (r'(?:budget|limit|max|at most|no more than)\s+\d+', 'limit'),
        (r'(?:file|path|dir|directory):\s*\S+', 'path'),
    ]
    for pattern, _category in patterns:
        matches = re.findall(pattern, goal, re.IGNORECASE)
        for m in matches[:2]:  # Cap at 2 per category
            cleaned = m.strip()
            if len(cleaned) > 10 and cleaned not in constraints:
                constraints.append(cleaned)
    return constraints[:8]  # Max 8 constraints total


def _augment_state_summary(
    context: StepContext[ToolUseState],
    *,
    nag_after_rounds: int,
    skills: Optional['SkillLoader'] = None,
    task_store: TaskStore | None = None,
    compression_config: CompactConfig | None = None,
    background_runner: BackgroundCommandRunner | None = None,
) -> Dict[str, object]:
    summary = dict(context.state_summary)
    summary['todo_state'] = _build_todo_state_summary(context.state, nag_after_rounds=nag_after_rounds)
    summary['task_state'] = _build_task_state_summary(task_store)
    config = compression_config or CompactConfig()
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
    if background_runner is not None:
        summary['background_tasks'] = [item.to_dict() for item in background_runner.snapshot()]
    else:
        summary['background_tasks'] = []
    summary['notification_queue'] = [
        {'id': item.id, 'ok': item.ok, 'output': item.output[:500], 'error': item.error}
        for item in context.state.background_notifications
    ]
    reminder = summary['todo_state'].get('reminder')
    if reminder:
        summary['todo_reminder'] = reminder
    if skills is not None:
        summary['available_skills'] = skills.metadata()
    # Extract key constraints from goal to prevent context drift
    constraints = _extract_key_constraints(context.goal)
    if constraints:
        summary['key_constraints'] = constraints
    return summary


def _dispatch_tool_calls(
    *,
    tool_context: ToolContext,
    dispatch_map: ToolDispatchMap,
    tool_calls,
    hook_manager: HookManager | None = None,
    loop_detector: LoopDetector | None = None,
    session_id: str = '',
    workspace_root: str = '',
    parallel: bool = True,
) -> List[ToolResult]:
    """Dispatch tool calls, optionally executing independent tools in parallel.

    When *parallel* is True (default), tools that pass hooks are executed
    concurrently via a thread pool — matching pi-mono's Promise.all pattern.
    Read-only tools (read_file, search, etc.) benefit most; mutating tools
    still run but the GIL keeps them safe for file I/O.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Phase 1: Loop detection + PreToolUse hooks (sequential, per-tool)
    ready: list[tuple[int, object, ToolResult | None]] = []  # (index, tool_call, pre_result_or_None)
    for idx, tool_call in enumerate(tool_calls):
        # --- Loop detection ---
        if loop_detector is not None:
            loop_msg = loop_detector.check(tool_call.name, tool_call.arguments)
            if loop_msg:
                ready.append((idx, tool_call, ToolResult(
                    id=tool_call.id, ok=False, output='', error=loop_msg,
                )))
                continue

        # --- PreToolUse hook ---
        if hook_manager is not None and hook_manager.has_hooks(HookEvent.PreToolUse):
            hook_input = build_hook_input_for_tool(
                HookEvent.PreToolUse,
                tool_call.name,
                tool_call.arguments,
                session_id=session_id,
                workspace_root=workspace_root,
            )
            hook_result = hook_manager.run_event(HookEvent.PreToolUse, hook_input)
            if not hook_result.approved:
                ready.append((idx, tool_call, ToolResult(
                    id=tool_call.id, ok=False, output='',
                    error=f'blocked by hook: {hook_result.error}',
                )))
                continue
            # Apply modified input from hook if provided
            if hook_result.modified_input:
                tool_call = type(tool_call)(
                    id=tool_call.id,
                    name=tool_call.name,
                    arguments=hook_result.modified_input,
                )

        ready.append((idx, tool_call, None))  # None = needs execution

    # Phase 2: Execute tools (parallel or sequential)
    def _exec_one(tc):
        return execute_tool_call(tool_context, tc, dispatch_map)

    exec_indices = [i for i, (idx, tc, pre) in enumerate(ready) if pre is None]
    results_by_pos: dict[int, ToolResult] = {}

    if parallel and len(exec_indices) > 1:
        with ThreadPoolExecutor(max_workers=min(len(exec_indices), 8)) as pool:
            futures = {
                pool.submit(_exec_one, ready[i][1]): i
                for i in exec_indices
            }
            for future in as_completed(futures):
                pos = futures[future]
                try:
                    results_by_pos[pos] = future.result()
                except Exception as exc:
                    tc = ready[pos][1]
                    results_by_pos[pos] = ToolResult(
                        id=tc.id, ok=False, output='', error=f'parallel execution error: {exc}',
                    )
    else:
        for i in exec_indices:
            results_by_pos[i] = _exec_one(ready[i][1])

    # Phase 3: Assemble results + PostToolUse hooks (sequential, per-tool)
    executed: List[ToolResult] = []
    for i, (idx, tool_call, pre_result) in enumerate(ready):
        result = pre_result if pre_result is not None else results_by_pos[i]

        # --- PostToolUse hook ---
        if hook_manager is not None and hook_manager.has_hooks(HookEvent.PostToolUse):
            post_input = build_hook_input_for_tool(
                HookEvent.PostToolUse,
                tool_call.name,
                tool_call.arguments,
                tool_output=result.output if result.ok else (result.error or ''),
                session_id=session_id,
                workspace_root=workspace_root,
            )
            hook_manager.run_event(HookEvent.PostToolUse, post_input)

        executed.append(result)
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


def _apply_background_notifications(
    state: ToolUseState,
    notifications: Tuple[ToolResult, ...],
) -> ToolUseState:
    if not notifications:
        return state

    history = list(state.history)
    transcript = list(state.transcript)
    for item in notifications:
        status = 'ok' if item.ok else f'error={item.error}'
        history.append(f'notification[{item.id}] {status}')
        content = item.output if item.ok else (item.error or item.output or 'background task error')
        transcript.append(
            TranscriptEntry(
                kind='tool_result',
                content=content[:4000],
                tool_name='run_command_async',
                call_id=item.id,
                ok=item.ok,
            )
        )

    return state.replace(
        history=tuple(history),
        tool_results=notifications,
        transcript=tuple(transcript),
        background_notifications=notifications,
    )


def _build_round_metadata(
    *,
    context: StepContext[ToolUseState],
    state_summary: Dict[str, object],
    thought: str,
    plan: str,
    tool_calls,
    tool_results: List[ToolResult],
) -> Dict[str, object]:
    has_tool_calls = len(tool_calls) > 0
    return {
        'has_tool_calls': has_tool_calls,
        'thought': thought,
        'plan': plan,
        'tool_calls': [
            {'id': call.id, 'name': call.name, 'arguments': dict(call.arguments)}
            for call in tool_calls
        ],
        'tool_results': [
            {
                'id': item.id,
                'ok': item.ok,
                'error': item.error,
                'output': item.output[:2000],
                'permission_decision': item.metadata.get('permission_decision'),
                'permission_reason': item.metadata.get('permission_reason'),
            }
            for item in tool_results
        ],
        'state_summary': state_summary,
        'last_steps': list(context.last_steps),
    }


def _append_transcript_entries(
    state: ToolUseState,
    *,
    thought: str,
    tool_calls,
    tool_results: List[ToolResult],
    now_s: float = 0.0,
) -> Tuple[TranscriptEntry, ...]:
    _now = now_s if now_s > 0 else _time.time()
    entries = list(state.transcript)
    entries.append(TranscriptEntry(kind='thought', content=thought, created_at=_now))
    for call, result in zip(tool_calls, tool_results):
        content = result.output if result.ok else (result.error or result.output or 'tool error')
        entries.append(
            TranscriptEntry(
                kind='tool_result',
                content=content[:4000],
                tool_name=call.name,
                call_id=result.id,
                ok=result.ok,
                created_at=_now,
            )
        )
    return tuple(entries)


def _compact_state_if_needed(
    *,
    goal: str,
    state: ToolUseState,
    transcripts_dir: Path | None,
    summarizer: SummarizerFn | None,
    compression_config: CompactConfig,
    compact_manager: CompactManager,
) -> ToolUseState:
    compacted_transcript = micro_compact_entries(
        state.transcript,
        keep_last_results=compression_config.micro_keep_last_results,
    )
    # Time-based microcompact: clear ALL tool results after long inactivity gap
    compacted_transcript = time_based_micro_compact(compacted_transcript)
    next_state = state.replace(transcript=compacted_transcript)

    estimated_tokens = estimate_tokens(
        [next_state.compact_summary, *[entry.content for entry in next_state.transcript]]
    )
    reason = ''
    if compact_manager.requested:
        reason = compact_manager.reason or 'manual'
    elif estimated_tokens > compression_config.max_context_tokens:
        reason = f'auto:{estimated_tokens}>{compression_config.max_context_tokens}'
    elif estimated_tokens > compression_config.max_context_tokens * compression_config.warn_tokens_percent:
        reason = f'auto:warn:{estimated_tokens}>{int(compression_config.warn_tokens_percent * 100)}%'

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
    return next_state.replace(
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
    compression_config: CompactConfig | None = None,
    transcripts_dir: Path | None = None,
    summarizer: SummarizerFn | None = None,
    hook_manager: HookManager | None = None,
    loop_detector: LoopDetector | None = None,
) -> StepResult[ToolUseState]:
    config = compression_config or CompactConfig()
    config.validate()
    background_runner = tool_context.background_runner
    notifications = background_runner.drain_notifications() if background_runner is not None else tuple()
    effective_state = _apply_background_notifications(context.state, notifications)
    effective_context = StepContext(
        goal=context.goal,
        state=effective_state,
        step_index=context.step_index,
        started_at_s=context.started_at_s,
        now_s=context.now_s,
        history=context.history,
        state_summary=context.state_summary,
        last_steps=context.last_steps,
    )
    augmented_state_summary = _augment_state_summary(
        effective_context,
        nag_after_rounds=nag_after_rounds,
        skills=skills,
        task_store=task_store,
        compression_config=config,
        background_runner=background_runner,
    )
    augmented_state_summary.setdefault('workspace', {'root': str(tool_context.workspace_root)})
    # Inject Reflexion: self-critique after tool failures
    reflection = _build_reflection(list(effective_state.tool_results))
    if reflection:
        augmented_state_summary['reflection'] = reflection
    # Inject loop detector advisory if needed
    if loop_detector is not None:
        augmented_state_summary['loop_detector'] = {'max_repeats': loop_detector.max_repeats}
    todo_manager = TodoManager(
        TodoSnapshot(
            items=effective_state.todos,
            rounds_since_update=effective_state.rounds_since_todo_update,
        )
    )
    tool_context = ToolContext(
        workspace_root=tool_context.workspace_root,
        policy=tool_context.policy,
        todo_manager=todo_manager,
        skill_loader=skills,
        compact_manager=CompactManager(),
        background_runner=background_runner,
    )
    raw = _decide_next_step(decider, effective_context, augmented_state_summary)
    parsed = parse_agent_step(raw)
    if parsed is None:
        output = 'invalid agent step json. expected schema: ' + render_agent_step_schema()
        return StepResult(
            output=output,
            state=effective_state,
            done=False,
            metadata={'parse_error': True, 'raw_response': raw[:2000]},
        )

    executed = _dispatch_tool_calls(
        tool_context=tool_context,
        dispatch_map=dispatch_map,
        tool_calls=parsed.tool_calls,
        hook_manager=hook_manager,
        loop_detector=loop_detector,
        session_id=str(tool_context.workspace_root),
        workspace_root=str(tool_context.workspace_root),
    )
    updated_history = _append_tool_history(
        history=effective_state.history,
        thought=parsed.thought,
        tool_results=executed,
    )
    updated_transcript = _append_transcript_entries(
        effective_state,
        thought=parsed.thought,
        tool_calls=parsed.tool_calls,
        tool_results=executed,
    )
    todo_snapshot = todo_manager.snapshot(previous_rounds_since_update=effective_state.rounds_since_todo_update)
    draft_state = effective_state.replace(
        history=updated_history,
        tool_results=tuple(executed),
        todos=todo_snapshot.items,
        rounds_since_todo_update=todo_snapshot.rounds_since_update,
        transcript=updated_transcript,
        background_notifications=notifications,
    )
    compacted_state = _compact_state_if_needed(
        goal=effective_context.goal,
        state=draft_state,
        transcripts_dir=transcripts_dir,
        summarizer=summarizer,
        compression_config=config,
        compact_manager=tool_context.compact_manager or CompactManager(),
    )
    new_state = compacted_state
    metadata = _build_round_metadata(
        context=effective_context,
        state_summary=augmented_state_summary,
        thought=parsed.thought,
        plan=parsed.plan,
        tool_calls=parsed.tool_calls,
        tool_results=executed,
    )
    metadata['raw_response'] = raw[:2000]
    metadata['todo_state'] = _build_todo_state_summary(new_state, nag_after_rounds=nag_after_rounds)
    metadata['compression_state'] = {
        'summary': new_state.compact_summary,
        'compaction_count': new_state.compaction_count,
        'archived_transcripts': list(new_state.archived_transcripts[-5:]),
        'recent_transcript': [entry.render_line() for entry in new_state.transcript[-config.recent_transcript_entries :]],
        'last_compaction_reason': new_state.last_compaction_reason,
    }
    metadata['background_notifications'] = [
        {'id': item.id, 'ok': item.ok, 'output': item.output[:500], 'error': item.error}
        for item in notifications
    ]
    metadata['background_tasks'] = [
        item.to_dict() for item in background_runner.snapshot()
    ] if background_runner is not None else []
    if not metadata['has_tool_calls']:
        previous_failed_tools = [item for item in effective_state.tool_results if not item.ok]
        if previous_failed_tools:
            last_error = previous_failed_tools[-1].error or 'tool failed'
            return StepResult(
                output=f'tool action failed: {last_error}',
                state=new_state,
                done=False,
                metadata={**metadata, 'unresolved_tool_error': True},
            )
        if _looks_like_file_action(effective_context.goal) and not _has_successful_file_mutation(effective_state):
            return StepResult(
                output='tool action required: file operation requests must be completed with tool calls',
                state=new_state,
                done=False,
                metadata={**metadata, 'missing_tool_calls': True},
            )
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
    compression_config: CompactConfig | None = None,
    transcripts_dir: Path | None = None,
    summarizer: SummarizerFn | None = None,
    hook_manager: HookManager | None = None,
    loop_detector: LoopDetector | None = None,
) -> Callable[[StepContext[ToolUseState]], StepResult[ToolUseState]]:
    dispatch_map = build_tool_dispatch(skills=skills, extra_tools=extra_tools)
    tool_context = ToolContext(
        workspace_root=workspace_root,
        policy=policy,
        background_runner=BackgroundCommandRunner(workspace_root),
    )

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
            hook_manager=hook_manager,
            loop_detector=loop_detector,
        )

    return step
