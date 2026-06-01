from __future__ import annotations

from ..messages import render_transcript
from ..session import SessionStore
from .event_viewer import render_event_stream

__all__ = ['render_session', 'render_session_diff', 'parse_limit']


def parse_limit(argument: str, *, default: int, maximum: int) -> int:
    raw = argument.strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, min(value, maximum))


def render_summary_text(session_store: SessionStore) -> str:
    text = session_store.state.last_summary.strip()
    return f'summary:\n{text}' if text else 'summary:\n(empty)'


def render_status_summary(session_store: SessionStore) -> str:
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


def render_history_summary(session_store: SessionStore, *, limit: int = 8) -> str:
    transcript = render_transcript(session_store.state.history_tail[-limit:])
    return f'recent_history:\n{transcript}'


def render_event_summary(session_store: SessionStore, *, limit: int = 10) -> str:
    return 'recent_events:\n' + render_event_stream(session_store.events_file, limit=limit)


def render_permission_summary(session_store: SessionStore) -> str:
    stats = session_store.state.permission_stats
    cache_size = len(session_store.state.permission_cache)
    return (
        'permissions:\n'
        f'allow: {stats.get("allow", 0)}\n'
        f'deny: {stats.get("deny", 0)}\n'
        f'ask: {stats.get("ask", 0)}\n'
        f'cached_rules: {cache_size}'
    )


def render_todo_summary(session_store: SessionStore) -> str:
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


def render_session_panel(session_store: SessionStore, *, history_limit: int = 5, event_limit: int = 5) -> str:
    return (
        render_status_summary(session_store)
        + '\n\n'
        + render_summary_text(session_store)
        + '\n\n'
        + render_history_summary(session_store, limit=history_limit)
        + '\n\n'
        + render_event_summary(session_store, limit=event_limit)
        + '\n\n'
        + render_permission_summary(session_store)
        + '\n\n'
        + render_todo_summary(session_store)
    )
