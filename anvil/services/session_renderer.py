from __future__ import annotations

from ..messages import render_transcript
from ..runtime.session import SessionStore
from .event_viewer import render_event_stream

__all__ = ['render_session', 'render_session_diff', 'parse_limit', 'render_cache_summary', 'render_rate_limit_summary', 'render_loop_stats']


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


def render_cache_summary(usage_tracker) -> str:
    """Render prompt-cache hit statistics from a TokenUsageTracker."""
    s = usage_tracker.summary()
    calls = s['calls']
    if calls == 0:
        return 'cache_stats:\n(no API calls yet)'
    hit_rate = s['cache_hit_rate']
    cache_read = s['cache_read_tokens']
    cache_write = s['cache_creation_tokens']
    total_in = s['input_tokens']
    total_out = s['output_tokens']
    savings = s['estimated_cost_savings']
    lines = [
        'cache_stats:',
        f'  calls: {calls}',
        f'  input_tokens: {total_in}',
        f'  output_tokens: {total_out}',
        f'  cache_write_tokens: {cache_write}',
        f'  cache_read_tokens: {cache_read}',
        f'  cache_hit_rate: {hit_rate:.1%}',
        f'  savings: {savings["savings_percent"]:.1f}% (vs no-cache baseline)',
    ]
    return '\n'.join(lines)


def render_rate_limit_summary(rate_limit_tracker) -> str:
    """Render rate limit state from a RateLimitTracker."""
    s = rate_limit_tracker.summary()
    if not s['has_data']:
        return 'rate_limits:\n(no data yet)'
    lines = ['rate_limits:']
    if s['remaining_requests'] is not None:
        lines.append(f'  remaining_requests: {s["remaining_requests"]}')
    if s['limit_requests'] is not None:
        lines.append(f'  limit_requests: {s["limit_requests"]}')
    if s['remaining_tokens'] is not None:
        lines.append(f'  remaining_tokens: {s["remaining_tokens"]}')
    if s['limit_tokens'] is not None:
        lines.append(f'  limit_tokens: {s["limit_tokens"]}')
    if s['reset_requests'] is not None:
        lines.append(f'  reset_requests: {s["reset_requests"]}')
    if s['reset_tokens'] is not None:
        lines.append(f'  reset_tokens: {s["reset_tokens"]}')
    return '\n'.join(lines) if len(lines) > 1 else 'rate_limits:\n(no data)'


def render_loop_stats(session_store: SessionStore) -> str:
    tool_history = session_store.state.tool_history
    if not tool_history:
        return 'loop_stats:\n(no tool calls yet)'
    total = len(tool_history)
    ok = sum(1 for item in tool_history if item.get('ok'))
    errors = total - ok
    lines = [
        'loop_stats:',
        f'  tool_calls: {total}',
        f'  ok: {ok}',
        f'  errors: {errors}',
    ]
    names: dict[str, int] = {}
    for item in tool_history:
        name = str(item.get('name') or 'unknown')
        names[name] = names.get(name, 0) + 1
    top = sorted(names.items(), key=lambda kv: kv[1], reverse=True)[:5]
    if top:
        lines.append('  top_tools:')
        for name, count in top:
            lines.append(f'    {name}: {count}')
    return '\n'.join(lines)


def render_status_summary(session_store: SessionStore, *, usage_tracker=None, rate_limit_tracker=None) -> str:
    state = session_store.state
    base = (
        f'session_id: {state.session_id}\n'
        f'workspace: {state.workspace_root}\n'
        f'goal: {state.goal or "(empty)"}\n'
        f'status: {state.status}\n'
        f'created_at: {state.created_at}\n'
        f'updated_at: {state.updated_at}\n'
        f'last_summary: {state.last_summary or "(empty)"}'
    )
    parts = [base]
    parts.append(render_loop_stats(session_store))
    if usage_tracker is not None:
        parts.append(render_cache_summary(usage_tracker))
    if rate_limit_tracker is not None:
        parts.append(render_rate_limit_summary(rate_limit_tracker))
    return '\n\n'.join(parts)


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


def render_session_panel(session_store: SessionStore, *, history_limit: int = 5, event_limit: int = 5, usage_tracker=None, rate_limit_tracker=None) -> str:
    return (
        render_status_summary(session_store, usage_tracker=usage_tracker, rate_limit_tracker=rate_limit_tracker)
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
