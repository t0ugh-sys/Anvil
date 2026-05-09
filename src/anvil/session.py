from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .run_schema import EventRow, SCHEMA_VERSION, utc_now_iso

MAX_HISTORY_TAIL = 20
MAX_TOOL_HISTORY = 50
TAIL_WINDOW_BYTES = 16 * 1024  # 16KB tail window for fast metadata reads
DIRTY_WRITE_INTERVAL_S = 5.0  # Write session state at most every 5s when dirty


def _default_session_id() -> str:
    return utc_now_iso().replace(':', '').replace('-', '').replace('+00:00', 'Z')


@dataclass
class SessionState:
    session_id: str
    workspace_root: str
    goal: str
    status: str
    created_at: str
    updated_at: str
    history_tail: list[str] = field(default_factory=list)
    tool_history: list[dict[str, object]] = field(default_factory=list)
    todo_state: dict[str, object] = field(default_factory=dict)
    permission_cache: dict[str, str] = field(default_factory=dict)
    memory_run_dir: str = ''
    artifacts_dir: str = ''
    last_summary: str = ''
    permission_stats: dict[str, int] = field(default_factory=lambda: {'allow': 0, 'deny': 0, 'ask': 0})

    def to_dict(self) -> Dict[str, object]:
        return {
            'session_id': self.session_id,
            'workspace_root': self.workspace_root,
            'goal': self.goal,
            'status': self.status,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'history_tail': list(self.history_tail),
            'tool_history': list(self.tool_history),
            'todo_state': dict(self.todo_state),
            'permission_cache': dict(self.permission_cache),
            'memory_run_dir': self.memory_run_dir,
            'artifacts_dir': self.artifacts_dir,
            'last_summary': self.last_summary,
            'permission_stats': dict(self.permission_stats),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> 'SessionState':
        now = utc_now_iso()
        return cls(
            session_id=str(payload.get('session_id', _default_session_id())),
            workspace_root=str(payload.get('workspace_root', '')),
            goal=str(payload.get('goal', '')),
            status=str(payload.get('status', 'active')),
            created_at=str(payload.get('created_at', now)),
            updated_at=str(payload.get('updated_at', now)),
            history_tail=[item for item in payload.get('history_tail', []) if isinstance(item, str)],
            tool_history=[item for item in payload.get('tool_history', []) if isinstance(item, dict)],
            todo_state=dict(payload.get('todo_state', {})) if isinstance(payload.get('todo_state', {}), dict) else {},
            permission_cache={
                str(key): str(value)
                for key, value in dict(payload.get('permission_cache', {})).items()
            } if isinstance(payload.get('permission_cache', {}), dict) else {},
            memory_run_dir=str(payload.get('memory_run_dir', '')),
            artifacts_dir=str(payload.get('artifacts_dir', '')),
            last_summary=str(payload.get('last_summary', '')),
            permission_stats={
                str(key): int(value)
                for key, value in dict(payload.get('permission_stats', {'allow': 0, 'deny': 0, 'ask': 0})).items()
                if str(key) in {'allow', 'deny', 'ask'}
            },
        )


class SessionStore:
    def __init__(self, root_dir: Path, state: SessionState) -> None:
        self.root_dir = root_dir
        self.state = state
        self.session_dir = self.root_dir / self.state.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.session_file = self.session_dir / 'session.json'
        self.events_file = self.session_dir / 'events.jsonl'
        self.summary_file = self.session_dir / 'summary.json'
        self._dirty = False
        self._last_write_time = 0.0
        self._write_session()
        if not self.summary_file.exists():
            self.write_summary({})

    @classmethod
    def create(
        cls,
        *,
        root_dir: Path,
        workspace_root: Path,
        goal: str,
        memory_run_dir: Path,
        artifacts_dir: str = '',
        session_id: Optional[str] = None,
    ) -> 'SessionStore':
        now = utc_now_iso()
        state = SessionState(
            session_id=session_id or _default_session_id(),
            workspace_root=str(workspace_root),
            goal=goal,
            status='active',
            created_at=now,
            updated_at=now,
            memory_run_dir=str(memory_run_dir),
            artifacts_dir=artifacts_dir,
        )
        return cls(root_dir=root_dir, state=state)

    @classmethod
    def load(cls, *, root_dir: Path, session_id: str) -> 'SessionStore':
        session_dir = root_dir / session_id
        session_file = session_dir / 'session.json'
        if not session_file.exists():
            raise FileNotFoundError(f'session not found: {session_id}')
        payload = json.loads(session_file.read_text(encoding='utf-8'))
        state = SessionState.from_dict(payload if isinstance(payload, dict) else {})
        return cls(root_dir=root_dir, state=state)

    def record_permission_cache(self, cache: Dict[str, str]) -> None:
        self.state.permission_cache = dict(cache)
        self.state.updated_at = utc_now_iso()
        self._write_session()

    def append_event(self, event: str, payload: Dict[str, Any]) -> None:
        tool_name, permission_decision, permission_reason = self._extract_event_annotations(payload)
        row = EventRow(
            schema_version=SCHEMA_VERSION,
            ts=utc_now_iso(),
            event=event,
            step=payload.get('step') if isinstance(payload.get('step'), int) else None,
            payload=payload,
            session_id=self.state.session_id,
            tool_name=tool_name,
            permission_decision=permission_decision,
            permission_reason=permission_reason,
        )
        with self.events_file.open('a', encoding='utf-8') as file:
            file.write(json.dumps(row.to_dict(), ensure_ascii=False))
            file.write('\n')
        self._update_state_from_event(event, payload)

    def write_summary(self, payload: Dict[str, Any]) -> None:
        merged = dict(payload)
        merged['session_id'] = self.state.session_id
        merged['goal'] = merged.get('goal') or self.state.goal
        merged['last_summary'] = self.state.last_summary
        merged['permission_stats'] = dict(self.state.permission_stats)
        self.summary_file.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding='utf-8')

    def _update_state_from_event(self, event: str, payload: Dict[str, Any]) -> None:
        self.state.updated_at = utc_now_iso()
        if event == 'run_started':
            goal = payload.get('goal')
            if isinstance(goal, str) and goal.strip():
                self.state.goal = goal
                self.state.status = 'active'
        if event in {'chat_user', 'chat_assistant'}:
            content = payload.get('content')
            role = payload.get('role')
            if isinstance(content, str) and content:
                prefix = f'{role}: ' if isinstance(role, str) and role else ''
                self.state.history_tail.append(prefix + content)
                self.state.history_tail = self.state.history_tail[-MAX_HISTORY_TAIL:]
                self.state.status = 'active'
        if event == 'step_succeeded':
            output = payload.get('output')
            if isinstance(output, str) and output:
                self.state.history_tail.append(output)
                self.state.history_tail = self.state.history_tail[-MAX_HISTORY_TAIL:]
            metadata = payload.get('metadata', {})
            if isinstance(metadata, dict):
                todo_state = metadata.get('todo_state')
                if isinstance(todo_state, dict):
                    self.state.todo_state = dict(todo_state)
                compression_state = metadata.get('compression_state')
                if isinstance(compression_state, dict):
                    self.state.last_summary = str(compression_state.get('summary', self.state.last_summary))
                tool_calls = metadata.get('tool_calls', [])
                tool_results = metadata.get('tool_results', [])
                call_names = {
                    str(item.get('id')): str(item.get('name'))
                    for item in tool_calls
                    if isinstance(item, dict) and isinstance(item.get('id'), str)
                }
                for result in tool_results:
                    if not isinstance(result, dict):
                        continue
                    call_id = str(result.get('id', ''))
                    self.state.tool_history.append(
                        {
                            'id': call_id,
                            'name': call_names.get(call_id, ''),
                            'ok': bool(result.get('ok', False)),
                            'error': result.get('error'),
                            'permission_decision': result.get('permission_decision'),
                            'permission_reason': result.get('permission_reason'),
                        }
                    )
                    self.state.tool_history = self.state.tool_history[-MAX_TOOL_HISTORY:]
                    mode = result.get('permission_decision')
                    if isinstance(mode, str) and mode in self.state.permission_stats:
                        self.state.permission_stats[mode] = self.state.permission_stats.get(mode, 0) + 1
        if event == 'run_finished':
            self.state.status = 'completed' if payload.get('done') else 'stopped'
        self._write_session()

    def _extract_event_annotations(self, payload: Dict[str, Any]) -> tuple[str | None, str | None, str | None]:
        metadata = payload.get('metadata', {})
        if not isinstance(metadata, dict):
            return None, None, None
        tool_calls = metadata.get('tool_calls', [])
        tool_results = metadata.get('tool_results', [])
        if len(tool_calls) != 1 or len(tool_results) != 1:
            return None, None, None
        call = tool_calls[0]
        result = tool_results[0]
        if not isinstance(call, dict) or not isinstance(result, dict):
            return None, None, None
        tool_name = call.get('name')
        decision = result.get('permission_decision')
        reason = result.get('permission_reason')
        return (
            str(tool_name) if isinstance(tool_name, str) else None,
            str(decision) if isinstance(decision, str) else None,
            str(reason) if isinstance(reason, str) else None,
        )

    def _write_session(self) -> None:
        self.session_file.write_text(json.dumps(self.state.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
        self._dirty = False
        self._last_write_time = time.monotonic()

    def mark_dirty(self) -> None:
        """Mark session state as needing a write."""
        self._dirty = True

    def flush_if_dirty(self) -> bool:
        """Write session state if dirty and enough time has passed.

        Returns True if a write was performed.
        """
        if not self._dirty:
            return False
        now = time.monotonic()
        if now - self._last_write_time < DIRTY_WRITE_INTERVAL_S:
            return False
        self._write_session()
        return True

    def force_flush(self) -> None:
        """Force write session state regardless of dirty/timing."""
        if self._dirty:
            self._write_session()

    @classmethod
    def load_fast(cls, *, root_dir: Path, session_id: str) -> 'SessionStore':
        """Fast session load using JSONL tail window.

        Instead of reading the full session.json, reads only the last 16KB
        of events.jsonl to reconstruct recent state. Falls back to full
        load if the tail window is insufficient.
        """
        session_dir = root_dir / session_id
        session_file = session_dir / 'session.json'
        events_file = session_dir / 'events.jsonl'

        # Try tail-window fast path
        if events_file.exists():
            try:
                state = cls._rebuild_from_tail(events_file, session_id)
                if state is not None:
                    store = cls.__new__(cls)
                    store.root_dir = root_dir
                    store.state = state
                    store.session_dir = session_dir
                    store.session_file = session_file
                    store.events_file = events_file
                    store.summary_file = session_dir / 'summary.json'
                    store._dirty = False
                    store._last_write_time = time.monotonic()
                    return store
            except Exception:
                pass  # Fall through to full load

        # Fallback: full load from session.json
        return cls.load(root_dir=root_dir, session_id=session_id)

    @classmethod
    def _rebuild_from_tail(cls, events_file: Path, session_id: str) -> SessionState | None:
        """Rebuild session state from the tail of the JSONL events file.

        Reads the last TAIL_WINDOW_BYTES to get recent events, then
        reconstructs the state. Returns None if insufficient data.
        """
        file_size = events_file.stat().st_size
        if file_size == 0:
            return None

        # Read tail window
        read_size = min(file_size, TAIL_WINDOW_BYTES)
        with events_file.open('rb') as f:
            if file_size > read_size:
                f.seek(file_size - read_size)
            raw = f.read(read_size).decode('utf-8', errors='replace')

        # Parse JSONL lines (skip partial first line if we seeked)
        lines = raw.split('\n')
        if file_size > read_size and lines:
            lines = lines[1:]  # Skip potentially partial first line

        events: List[Dict[str, Any]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        if not events:
            return None

        # Rebuild state from events
        now = utc_now_iso()
        state = SessionState(
            session_id=session_id,
            workspace_root='',
            goal='',
            status='active',
            created_at=now,
            updated_at=now,
        )

        for event in events:
            event_name = event.get('event', '')
            payload = event.get('payload', {})
            if not isinstance(payload, dict):
                continue

            if event_name == 'run_started':
                goal = payload.get('goal')
                if isinstance(goal, str):
                    state.goal = goal

            if event_name in {'chat_user', 'chat_assistant'}:
                content = payload.get('content')
                role = payload.get('role')
                if isinstance(content, str) and content:
                    prefix = f'{role}: ' if isinstance(role, str) and role else ''
                    state.history_tail.append(prefix + content)
                    state.history_tail = state.history_tail[-MAX_HISTORY_TAIL:]

            if event_name == 'step_succeeded':
                metadata = payload.get('metadata', {})
                if isinstance(metadata, dict):
                    todo = metadata.get('todo_state')
                    if isinstance(todo, dict):
                        state.todo_state = dict(todo)
                    comp = metadata.get('compression_state')
                    if isinstance(comp, dict):
                        state.last_summary = str(comp.get('summary', ''))

            if event_name == 'run_finished':
                state.status = 'completed' if payload.get('done') else 'stopped'

        return state
