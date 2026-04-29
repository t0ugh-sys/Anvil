from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..session import SessionState, SessionStore
from ..tool_spec import ToolSpec
from ..tool_views import render_tool_overview


@dataclass(frozen=True)
class SlashCommand:
    name: str
    argument: str = ''


@dataclass(frozen=True)
class CommandResult:
    output: str
    should_continue: bool = True


def render_session_header(
    state: SessionState,
    *,
    runtime_label: str = 'interactive',
) -> str:
    return (
        f'Anvil {runtime_label} session {state.session_id}\n'
        f'workspace: {state.workspace_root}\n'
        f'turns={state.turn_count} messages={state.message_count} '
        f'commands={state.command_count} steps={state.step_count}'
    )


def render_runtime_panel(
    state: SessionState,
    *,
    runtime_config_text: str,
) -> str:
    return (
        f'{render_session_header(state, runtime_label="runtime")}\n'
        f'status: {state.status}\n'
        f'goal: {state.goal or "(empty)"}\n'
        f'last_activity_at: {state.last_activity_at or state.updated_at}\n'
        f'last_stop_reason: {state.last_stop_reason or "(empty)"}\n'
        f'permission_mode_cache_entries: {len(state.permission_cache)}\n'
        f'memory_run_dir: {state.memory_run_dir or "(empty)"}\n'
        f'artifacts_dir: {state.artifacts_dir or "(empty)"}\n'
        f'{runtime_config_text}'
    )


def render_session_status(
    state: SessionState,
    *,
    runtime_config_text: str,
) -> str:
    history = '\n'.join(state.history_tail[-5:]) if state.history_tail else '(empty)'
    summary = state.last_summary or '(empty)'
    todo_lines = state.todo_state.get('lines', []) if isinstance(state.todo_state, dict) else []
    if isinstance(todo_lines, list) and todo_lines:
        todo_text = '\n'.join(str(item) for item in todo_lines[-8:])
    else:
        todo_text = '(empty)'
    recent_tools = state.tool_history[-5:]
    if recent_tools:
        tool_lines = []
        for item in recent_tools:
            if not isinstance(item, dict):
                continue
            name = str(item.get('name') or '(unknown)')
            ok = 'ok' if item.get('ok') else 'error'
            permission = str(item.get('permission_decision') or '-')
            tool_lines.append(f'- {name} [{ok}] permission={permission}')
        tool_text = '\n'.join(tool_lines) if tool_lines else '(empty)'
    else:
        tool_text = '(empty)'
    return (
        f'session_id: {state.session_id}\n'
        f'workspace: {state.workspace_root}\n'
        f'goal: {state.goal or "(empty)"}\n'
        f'status: {state.status}\n'
        f'turn_count: {state.turn_count}\n'
        f'message_count: {state.message_count}\n'
        f'command_count: {state.command_count}\n'
        f'step_count: {state.step_count}\n'
        f'last_activity_at: {state.last_activity_at or state.updated_at}\n'
        f'last_user_message: {state.last_user_message or "(empty)"}\n'
        f'last_stop_reason: {state.last_stop_reason or "(empty)"}\n'
        f'last_blocked_tool: {state.last_blocked_tool_name or "(empty)"}\n'
        f'last_blocked_reason: {state.last_blocked_tool_reason or "(empty)"}\n'
        f'memory_run_dir: {state.memory_run_dir or "(empty)"}\n'
        f'artifacts_dir: {state.artifacts_dir or "(empty)"}\n'
        f'permission_stats: allow={state.permission_stats.get("allow", 0)} '
        f'deny={state.permission_stats.get("deny", 0)} ask={state.permission_stats.get("ask", 0)}\n'
        f'last_summary: {summary}\n'
        f'{runtime_config_text}\n'
        f'todo_state:\n{todo_text}\n'
        f'recent_tools:\n{tool_text}\n'
        f'history_tail:\n{history}'
    )


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
    runtime_config_manager=None,
) -> CommandResult:
    if command.name == 'help':
        return CommandResult(
            output=(
                'Commands:\n'
                '/help   Show this help\n'
                '/status Show current session status\n'
                '/config Show current runtime config\n'
                '/provider <name> Set provider for the current session\n'
                '/model <name> Set model for the current session\n'
                '/wire-api <name> Set wire API for the current session\n'
                '/base-url <url> Set base URL for the current session\n'
                '/tools  List available tools\n'
                '/resume Show the current session summary\n'
                '/exit   Exit the interactive runtime'
            )
        )
    if command.name == 'config':
        if runtime_config_manager is None:
            return CommandResult(output='runtime config unavailable')
        return CommandResult(
            output=render_runtime_panel(
                session_store.state,
                runtime_config_text=runtime_config_manager.summary(),
            )
        )
    if command.name == 'provider':
        if runtime_config_manager is None:
            return CommandResult(output='runtime config unavailable')
        try:
            return CommandResult(output=runtime_config_manager.set_provider(command.argument))
        except ValueError as error:
            return CommandResult(output=str(error))
    if command.name == 'model':
        if runtime_config_manager is None:
            return CommandResult(output='runtime config unavailable')
        try:
            return CommandResult(output=runtime_config_manager.set_model(command.argument))
        except ValueError as error:
            return CommandResult(output=str(error))
    if command.name == 'wire-api':
        if runtime_config_manager is None:
            return CommandResult(output='runtime config unavailable')
        try:
            return CommandResult(output=runtime_config_manager.set_wire_api(command.argument))
        except ValueError as error:
            return CommandResult(output=str(error))
    if command.name == 'base-url':
        if runtime_config_manager is None:
            return CommandResult(output='runtime config unavailable')
        return CommandResult(output=runtime_config_manager.set_base_url(command.argument))
    if command.name == 'tools':
        specs = sorted(tool_specs, key=lambda item: item.name)
        return CommandResult(output=render_tool_overview(specs) if specs else 'No tools registered.')
    if command.name in {'resume', 'status'}:
        state = session_store.state
        config_text = runtime_config_manager.summary() if runtime_config_manager is not None else 'runtime config unavailable'
        return CommandResult(output=render_session_status(state, runtime_config_text=config_text))
    if command.name == 'exit':
        return CommandResult(output='bye', should_continue=False)
    return CommandResult(output=f'Unknown command: /{command.name}')
