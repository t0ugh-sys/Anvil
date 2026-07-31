from __future__ import annotations

import json
import shlex
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from ..runtime.session import SessionStore
from ..services.session_renderer import (
    parse_limit as _parse_limit,
    render_cache_summary,
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


def format_status_summary(session_store: SessionStore, *, usage_tracker=None, rate_limit_tracker=None) -> str:
    return render_status_summary(session_store, usage_tracker=usage_tracker, rate_limit_tracker=rate_limit_tracker)


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


def _execute_gc_command(argument: str, *, session_store: SessionStore) -> CommandResult:
    try:
        tokens = shlex.split(argument)
    except ValueError:
        tokens = argument.split()

    dry_run = '--dry-run' in tokens
    keep_days = 30
    keep_count = 100

    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == '--keep-days' and i + 1 < len(tokens):
            try:
                keep_days = int(tokens[i + 1])
            except ValueError:
                pass
            i += 2
        elif t == '--keep-count' and i + 1 < len(tokens):
            try:
                keep_count = int(tokens[i + 1])
            except ValueError:
                pass
            i += 2
        else:
            i += 1

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=keep_days)

    sessions_dir = session_store.root_dir
    sessions_to_delete: list[Path] = []
    if sessions_dir.exists():
        for sd in sorted(sessions_dir.iterdir()):
            if not sd.is_dir():
                continue
            if sd.name == session_store.state.session_id:
                continue
            created_at = None
            session_file = sd / 'session.json'
            if session_file.exists():
                try:
                    payload = json.loads(session_file.read_text(encoding='utf-8'))
                    ts = payload.get('created_at', '')
                    if ts:
                        created_at = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                except Exception:
                    pass
            if created_at is None:
                created_at = datetime.fromtimestamp(sd.stat().st_mtime, tz=timezone.utc)
            if created_at < cutoff:
                sessions_to_delete.append(sd)

    runs_to_delete: list[Path] = []
    runs_dir_str = session_store.state.memory_run_dir
    if runs_dir_str:
        runs_dir = Path(runs_dir_str)
        if runs_dir.exists():
            all_runs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            runs_to_delete = list(all_runs[keep_count:])

    def _path_size(p: Path) -> int:
        if p.is_file():
            return p.stat().st_size
        total = 0
        for child in p.rglob('*'):
            if child.is_file():
                try:
                    total += child.stat().st_size
                except OSError:
                    pass
        return total

    total_bytes = sum(_path_size(p) for p in sessions_to_delete) + sum(_path_size(p) for p in runs_to_delete)
    total_mb = total_bytes / (1024 * 1024)

    prefix = '[dry-run] ' if dry_run else ''
    lines = [
        f'{prefix}将释放 {total_mb:.1f} MB，删除 {len(sessions_to_delete)} 个会话，{len(runs_to_delete)} 个 run'
    ]

    if dry_run:
        if sessions_to_delete:
            lines.append(f'Sessions to delete (older than {keep_days}d):')
            for p in sessions_to_delete[:10]:
                lines.append(f'  {p.name}')
            if len(sessions_to_delete) > 10:
                lines.append(f'  ... and {len(sessions_to_delete) - 10} more')
        if runs_to_delete:
            lines.append(f'Runs to delete (keep last {keep_count}):')
            for p in runs_to_delete[:10]:
                lines.append(f'  {p.name}')
            if len(runs_to_delete) > 10:
                lines.append(f'  ... and {len(runs_to_delete) - 10} more')
        if not sessions_to_delete and not runs_to_delete:
            lines.append('Nothing to clean up.')
    else:
        for p in sessions_to_delete:
            shutil.rmtree(p, ignore_errors=True)
        for p in runs_to_delete:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    p.unlink()
                except OSError:
                    pass
        lines.append('Done.' if (sessions_to_delete or runs_to_delete) else 'Nothing to clean up.')

    return CommandResult(output='\n'.join(lines))


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
    usage_tracker=None,
    rate_limit_tracker=None,
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
                '/gc     Garbage-collect old sessions and runs\n'
                '/exit   Exit the interactive runtime'
            )
        )
    if command.name == 'status':
        return CommandResult(output=format_status_summary(session_store, usage_tracker=usage_tracker, rate_limit_tracker=rate_limit_tracker))
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
    if command.name == 'gc':
        return _execute_gc_command(command.argument, session_store=session_store)
    if command.name == 'pricing':
        return _execute_pricing_command(command.argument)
    if command.name == 'exit':
        return CommandResult(output='bye', should_continue=False)
    return CommandResult(output=f'Unknown command: /{command.name}')
