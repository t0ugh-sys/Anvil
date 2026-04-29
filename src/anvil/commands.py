from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .session import SessionStore
from .tool_spec import ToolSpec


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
) -> CommandResult:
    if command.name == 'help':
        return CommandResult(
            output=(
                'Commands:\n'
                '/help   Show this help\n'
                '/tools  List available tools\n'
                '/resume Show the current session summary\n'
                '/exit   Exit the interactive runtime'
            )
        )
    if command.name == 'tools':
        names = [spec.name for spec in sorted(tool_specs, key=lambda item: item.name)]
        return CommandResult(output='\n'.join(names) if names else 'No tools registered.')
    if command.name == 'resume':
        state = session_store.state
        history = '\n'.join(state.history_tail[-5:]) if state.history_tail else '(empty)'
        summary = state.last_summary or '(empty)'
        return CommandResult(
            output=(
                f'session_id: {state.session_id}\n'
                f'workspace: {state.workspace_root}\n'
                f'goal: {state.goal or "(empty)"}\n'
                f'status: {state.status}\n'
                f'last_summary: {summary}\n'
                f'history_tail:\n{history}'
            )
        )
    if command.name == 'exit':
        return CommandResult(output='bye', should_continue=False)
    return CommandResult(output=f'Unknown command: /{command.name}')
