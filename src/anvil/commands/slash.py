from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..messages import render_transcript
from ..session import SessionStore
from ..tool_spec import ToolSpec


@dataclass(frozen=True)
class SlashCommand:
    name: str
    argument: str = ''


@dataclass(frozen=True)
class CommandResult:
    output: str
    should_continue: bool = True


def format_status_summary(session_store: SessionStore) -> str:
    state = session_store.state
    return (
        f'session_id: {state.session_id}\n'
        f'workspace: {state.workspace_root}\n'
        f'goal: {state.goal or "(empty)"}\n'
        f'status: {state.status}\n'
        f'created_at: {state.created_at}\n'
        f'updated_at: {state.updated_at}\n'
        f'last_summary: {state.last_summary or "(empty)"}'
    )


def format_history_summary(session_store: SessionStore, *, limit: int = 8) -> str:
    state = session_store.state
    transcript = render_transcript(state.history_tail[-limit:])
    return f'recent_history:\n{transcript}'


def format_permission_summary(session_store: SessionStore) -> str:
    state = session_store.state
    stats = state.permission_stats
    cache_size = len(state.permission_cache)
    return (
        'permissions:\n'
        f'allow: {stats.get("allow", 0)}\n'
        f'deny: {stats.get("deny", 0)}\n'
        f'ask: {stats.get("ask", 0)}\n'
        f'cached_rules: {cache_size}'
    )


def format_todo_summary(session_store: SessionStore) -> str:
    todo_state = session_store.state.todo_state
    items = todo_state.get('items', []) if isinstance(todo_state, dict) else []
    if not isinstance(items, list) or not items:
        return 'todo:\n(empty)'
    lines: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        content = str(item.get('content', '')).strip()
        status = str(item.get('status', '')).strip() or 'pending'
        if content:
            lines.append(f'- [{status}] {content}')
    return 'todo:\n' + ('\n'.join(lines) if lines else '(empty)')


def parse_slash_command(line: str) -> SlashCommand | None:
    text = line.strip()
    if not text.startswith('/'):
        return None
    parts = text[1:].split(None, 1)
    if not parts or not parts[0]:
        return None
    return SlashCommand(name=parts[0].lower(), argument=parts[1].strip() if len(parts) > 1 else '')


def execute_slash_command(
    command: SlashCommand,
    *,
    session_store: SessionStore,
    tool_specs: Iterable[ToolSpec],
) -> CommandResult:
    if command.name == 'help':
        return CommandResult(
            output=(
                'Commands:\n'
                '/help   Show this help\n'
                '/status Show the current session status\n'
                '/history Show recent transcript history\n'
                '/permissions Show permission decisions and cache stats\n'
                '/todo   Show the current todo state\n'
                '/tools  List available tools\n'
                '/resume Show the combined session recap\n'
                '/exit   Exit the interactive runtime'
            )
        )
    if command.name == 'status':
        return CommandResult(output=format_status_summary(session_store))
    if command.name == 'history':
        return CommandResult(output=format_history_summary(session_store))
    if command.name == 'permissions':
        return CommandResult(output=format_permission_summary(session_store))
    if command.name == 'todo':
        return CommandResult(output=format_todo_summary(session_store))
    if command.name == 'tools':
        names = [spec.name for spec in sorted(tool_specs, key=lambda item: item.name)]
        return CommandResult(output='\n'.join(names) if names else 'No tools registered.')
    if command.name == 'resume':
        return CommandResult(
            output=(
                format_status_summary(session_store)
                + '\n\n'
                + format_history_summary(session_store, limit=5)
                + '\n\n'
                + format_permission_summary(session_store)
                + '\n\n'
                + format_todo_summary(session_store)
            )
        )
    if command.name == 'exit':
        return CommandResult(output='bye', should_continue=False)
    return CommandResult(output=f'Unknown command: /{command.name}')
