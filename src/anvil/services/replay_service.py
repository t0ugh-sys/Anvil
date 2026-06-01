from __future__ import annotations

from pathlib import Path

from .event_viewer import render_event_stream

__all__ = ['resolve_events_file', 'render_replay']


def resolve_events_file(*, events_file: str, session_id: str, sessions_dir: str) -> Path:
    if session_id:
        return Path(sessions_dir) / session_id / 'events.jsonl'
    return Path(events_file)


def render_replay(*, events_file: Path, pretty: bool = False, limit: int | None = None) -> str:
    if pretty:
        return render_event_stream(events_file, limit=limit)
    return events_file.read_text(encoding='utf-8')
