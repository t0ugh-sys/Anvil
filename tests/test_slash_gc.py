from __future__ import annotations

import json
import shutil
import time
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from anvil.commands import execute_slash_command, parse_slash_command
from anvil.runtime.session import SessionStore
from anvil.tools import builtin_tool_specs


class GcSlashCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path('tests/.tmp') / f'gc-{uuid.uuid4().hex}'
        self.sessions_dir = self.tmp_dir / 'sessions'
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_session_store(self) -> SessionStore:
        return SessionStore.create(
            root_dir=self.sessions_dir,
            workspace_root=self.tmp_dir,
            goal='gc test',
            memory_run_dir=self.tmp_dir / 'runs',
        )

    def _write_old_session(self, name: str, *, days_old: int) -> Path:
        session_dir = self.sessions_dir / name
        session_dir.mkdir(parents=True, exist_ok=True)
        created_at = (datetime.now(tz=timezone.utc) - timedelta(days=days_old)).isoformat()
        payload = {
            'session_id': name,
            'workspace_root': str(self.tmp_dir),
            'goal': 'old session',
            'status': 'stopped',
            'created_at': created_at,
            'updated_at': created_at,
        }
        (session_dir / 'session.json').write_text(json.dumps(payload), encoding='utf-8')
        return session_dir

    def test_should_dry_run_without_deleting(self) -> None:
        store = self._make_session_store()
        old_dir = self._write_old_session('old-session', days_old=45)

        result = execute_slash_command(
            parse_slash_command('/gc --dry-run'),
            session_store=store,
            tool_specs=builtin_tool_specs(),
        )

        self.assertIn('dry-run', result.output)
        self.assertIn('old-session', result.output)
        self.assertTrue(old_dir.exists())

    def test_should_delete_sessions_older_than_keep_days(self) -> None:
        store = self._make_session_store()
        old_dir = self._write_old_session('old-session', days_old=45)
        current_session_dir = store.session_dir

        result = execute_slash_command(
            parse_slash_command('/gc --keep-days 30'),
            session_store=store,
            tool_specs=builtin_tool_specs(),
        )

        self.assertFalse(old_dir.exists())
        self.assertTrue(current_session_dir.exists())
        self.assertIn('Done', result.output)

    def test_should_never_delete_current_session(self) -> None:
        store = self._make_session_store()
        # Force the current session's own created_at far in the past.
        store.state.created_at = (datetime.now(tz=timezone.utc) - timedelta(days=999)).isoformat()
        store.force_flush()

        execute_slash_command(
            parse_slash_command('/gc --keep-days 1'),
            session_store=store,
            tool_specs=builtin_tool_specs(),
        )

        self.assertTrue(store.session_dir.exists())

    def test_should_keep_recent_sessions_within_keep_days(self) -> None:
        store = self._make_session_store()
        recent_dir = self._write_old_session('recent-session', days_old=1)

        execute_slash_command(
            parse_slash_command('/gc --keep-days 30'),
            session_store=store,
            tool_specs=builtin_tool_specs(),
        )

        self.assertTrue(recent_dir.exists())

    def test_should_respect_keep_count_for_runs(self) -> None:
        store = self._make_session_store()
        runs_dir = Path(store.state.memory_run_dir)
        runs_dir.mkdir(parents=True, exist_ok=True)
        run_paths = []
        for i in range(5):
            p = runs_dir / f'run-{i}.json'
            p.write_text('{}', encoding='utf-8')
            run_paths.append(p)
            time.sleep(0.05)  # ensure distinct mtimes on all filesystems

        execute_slash_command(
            parse_slash_command('/gc --keep-count 2'),
            session_store=store,
            tool_specs=builtin_tool_specs(),
        )

        remaining = sorted(runs_dir.iterdir())
        self.assertEqual(len(remaining), 2)
        self.assertIn(run_paths[-1], remaining)
        self.assertIn(run_paths[-2], remaining)

    def test_should_report_nothing_to_clean_up(self) -> None:
        store = self._make_session_store()

        result = execute_slash_command(
            parse_slash_command('/gc --dry-run'),
            session_store=store,
            tool_specs=builtin_tool_specs(),
        )

        self.assertIn('Nothing to clean up', result.output)


if __name__ == '__main__':
    unittest.main()
