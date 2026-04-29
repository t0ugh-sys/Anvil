from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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
        return CommandResult(output=runtime_config_manager.summary())
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
        names = [spec.name for spec in sorted(tool_specs, key=lambda item: item.name)]
        return CommandResult(output='\n'.join(names) if names else 'No tools registered.')
    if command.name == 'resume':
        state = session_store.state
        history = '\n'.join(state.history_tail[-5:]) if state.history_tail else '(empty)'
        summary = state.last_summary or '(empty)'
        config_text = runtime_config_manager.summary() if runtime_config_manager is not None else 'runtime config unavailable'
        return CommandResult(
            output=(
                f'session_id: {state.session_id}\n'
                f'workspace: {state.workspace_root}\n'
                f'goal: {state.goal or "(empty)"}\n'
                f'status: {state.status}\n'
                f'last_summary: {summary}\n'
                f'{config_text}\n'
                f'history_tail:\n{history}'
            )
        )
    if command.name == 'exit':
        return CommandResult(output='bye', should_continue=False)
    return CommandResult(output=f'Unknown command: /{command.name}')
