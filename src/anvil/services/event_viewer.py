from __future__ import annotations

import json
from pathlib import Path


def load_event_rows(events_file: Path, *, limit: int | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not events_file.exists():
        return rows

    if limit is not None and limit > 0:
        # Tail read: read last N lines efficiently
        lines = _tail_lines(events_file, limit)
    else:
        lines = events_file.read_text(encoding='utf-8').splitlines()

    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _tail_lines(path: Path, n: int) -> list[str]:
    """Read last n lines from a file efficiently."""
    try:
        with path.open('rb') as f:
            # Seek to end and read backwards
            f.seek(0, 2)
            file_size = f.tell()
            if file_size == 0:
                return []

            # Read chunks from end until we have enough lines
            chunk_size = min(8192, file_size)
            lines: list[bytes] = []
            pos = file_size

            while pos > 0 and len(lines) < n + 1:
                read_size = min(chunk_size, pos)
                pos -= read_size
                f.seek(pos)
                chunk = f.read(read_size)
                lines = chunk.split(b'\n') + lines

            # Remove partial first line if we didn't read from start
            if pos > 0 and lines:
                lines = lines[1:]

            # Take last n lines
            result = []
            for line in lines[-n:]:
                try:
                    result.append(line.decode('utf-8'))
                except UnicodeDecodeError:
                    continue
            return result
    except Exception:
        # Fallback to full read
        return events_file.read_text(encoding='utf-8').splitlines()[-n:]


def render_event_row(row: dict[str, object]) -> str:
    ts = str(row.get('ts', '') or '')
    event = str(row.get('event', 'unknown'))
    session_id = str(row.get('session_id', '') or '')
    tool_name = str(row.get('tool_name', '') or '')
    permission = str(row.get('permission_decision', '') or '')
    parts = [part for part in [ts, event] if part]
    line = ' '.join(parts)
    if tool_name:
        line += f' [{tool_name}]'
    if permission:
        line += f' permission={permission}'
    if session_id:
        line += f' session={session_id}'
    return line or 'unknown'


def render_event_stream(events_file: Path, *, limit: int | None = None) -> str:
    rows = load_event_rows(events_file, limit=limit)
    rendered = [f'- {render_event_row(row)}' for row in rows]
    return '\n'.join(rendered) if rendered else '(empty)'
