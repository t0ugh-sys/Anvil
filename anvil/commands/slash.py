from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from ..runtime.session import SessionStore
from ..services.session_renderer import (
    parse_limit as _parse_limit,
    render_event_summary,
    render_history_summary,
    render_permission_summary,
    render_session_panel,
    render_status_summary,
    render_summary_text,
    render_todo_summary,
)
from ..tool_spec import ToolSpec

__all__ = ['SlashCommand', 'CommandResult', 'parse_slash_command', 'execute_slash_command']


@dataclass(frozen=True)
class SlashCommand:
    name: str
    argument: str = ''


@dataclass(frozen=True)
class CommandResult:
    output: str
    should_continue: bool = True


def format_summary_text(session_store: SessionStore) -> str:
    return render_summary_text(session_store)


def format_status_summary(session_store: SessionStore) -> str:
    return render_status_summary(session_store)


def format_history_summary(session_store: SessionStore, *, limit: int = 8) -> str:
    return render_history_summary(session_store, limit=limit)


def format_event_summary(session_store: SessionStore, *, limit: int = 10) -> str:
    return render_event_summary(session_store, limit=limit)


def format_permission_summary(session_store: SessionStore) -> str:
    return render_permission_summary(session_store)


def format_todo_summary(session_store: SessionStore) -> str:
    return render_todo_summary(session_store)


def format_session_panel(session_store: SessionStore, *, history_limit: int = 5, event_limit: int = 5) -> str:
    return render_session_panel(session_store, history_limit=history_limit, event_limit=event_limit)


def parse_slash_command(line: str) -> SlashCommand | None:
    text = line.strip()
    if not text.startswith('/'):
        return None
    parts = text[1:].split(None, 1)
    if not parts or not parts[0]:
        return None
    return SlashCommand(name=parts[0].lower(), argument=parts[1].strip() if len(parts) > 1 else '')


def _execute_pricing_command(argument: str) -> CommandResult:
    from ..llm.usage import PRICING_FILE, _CLAUDE_PRICING, _load_pricing_from_file

    parts = argument.split()

    def _load_file_data() -> dict:
        data = _load_pricing_from_file()
        if data is None:
            return {'updated_at': '', 'models': dict(_CLAUDE_PRICING)}
        try:
            raw = json.loads(PRICING_FILE.read_text(encoding='utf-8'))
            return raw if isinstance(raw, dict) else {'updated_at': '', 'models': dict(_CLAUDE_PRICING)}
        except Exception:
            return {'updated_at': '', 'models': dict(_CLAUDE_PRICING)}

    if not parts:
        file_data = _load_file_data()
        return CommandResult(output=json.dumps(file_data, indent=2, ensure_ascii=False))

    if len(parts) == 5:
        model_key, input_s, output_s, cache_write_s, cache_read_s = parts
        try:
            entry = {
                'input': float(input_s),
                'output': float(output_s),
                'cache_write': float(cache_write_s),
                'cache_read': float(cache_read_s),
            }
        except ValueError:
            return CommandResult(
                output='Usage: /pricing <model> <input> <output> <cache_write> <cache_read>\n'
                       'All price values must be numbers (USD per million tokens).'
            )
        file_data = _load_file_data()
        file_data.setdefault('models', {})[model_key] = entry
        file_data['updated_at'] = datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')
        PRICING_FILE.write_text(json.dumps(file_data, indent=2), encoding='utf-8')
        return CommandResult(output=f'Updated pricing for {model_key!r}.')

    return CommandResult(
        output='Usage:\n'
               '  /pricing                                          Show current pricing table\n'
               '  /pricing <model> <input> <output> <cw> <cr>      Add or update a model entry\n'
               'Prices are in USD per million tokens.'
    )


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
                '/model  Show the current model\n'
                '/summary Show the current compressed session summary\n'
                '/history Show recent transcript history\n'
                '/events Show recent recorded session events\n'
                '/permissions Show permission decisions and cache stats\n'
                '/todo   Show the current todo state\n'
                '/tools  List available tools\n'
                '/panel  Show the full session panel\n'
                '/resume Show the combined session recap\n'
                '/pricing Show or update the LLM pricing table\n'
                '/exit   Exit the interactive runtime'
            )
        )
    if command.name == 'status':
        return CommandResult(output=format_status_summary(session_store))
    if command.name == 'summary':
        return CommandResult(output=format_summary_text(session_store))
    if command.name == 'history':
        limit = _parse_limit(command.argument, default=8, maximum=50)
        return CommandResult(output=format_history_summary(session_store, limit=limit))
    if command.name == 'events':
        limit = _parse_limit(command.argument, default=10, maximum=50)
        return CommandResult(output=format_event_summary(session_store, limit=limit))
    if command.name == 'permissions':
        return CommandResult(output=format_permission_summary(session_store))
    if command.name == 'todo':
        return CommandResult(output=format_todo_summary(session_store))
    if command.name == 'tools':
        names = [spec.name for spec in sorted(tool_specs, key=lambda item: item.name)]
        query = command.argument.strip().lower()
        if query:
            names = [name for name in names if query in name.lower()]
        return CommandResult(output='\n'.join(names) if names else 'No tools registered.')
    if command.name == 'panel':
        return CommandResult(output=format_session_panel(session_store))
    if command.name == 'resume':
        return CommandResult(output=format_session_panel(session_store, history_limit=10, event_limit=10))
    if command.name == 'pricing':
        return _execute_pricing_command(command.argument)
    if command.name == 'exit':
        return CommandResult(output='bye', should_continue=False)
    return CommandResult(output=f'Unknown command: /{command.name}')
