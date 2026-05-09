from __future__ import annotations

import json
import time
import unittest
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401

from anvil.session import SessionStore, SessionState


class DirtyTrackingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path('tests/.tmp') / f'session-{uuid.uuid4().hex}'
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_should_track_dirty_state(self) -> None:
        store = SessionStore.create(
            root_dir=self.tmp_dir,
            workspace_root=Path('.'),
            goal='test',
            memory_run_dir=self.tmp_dir / 'mem',
        )
        self.assertFalse(store._dirty)
        store.mark_dirty()
        self.assertTrue(store._dirty)

    def test_should_flush_when_dirty(self) -> None:
        store = SessionStore.create(
            root_dir=self.tmp_dir,
            workspace_root=Path('.'),
            goal='test',
            memory_run_dir=self.tmp_dir / 'mem',
        )
        store._last_write_time = 0  # Force time gap
        store.mark_dirty()
        result = store.flush_if_dirty()
        self.assertTrue(result)
        self.assertFalse(store._dirty)

    def test_should_not_flush_when_not_dirty(self) -> None:
        store = SessionStore.create(
            root_dir=self.tmp_dir,
            workspace_root=Path('.'),
            goal='test',
            memory_run_dir=self.tmp_dir / 'mem',
        )
        result = store.flush_if_dirty()
        self.assertFalse(result)

    def test_should_not_flush_within_interval(self) -> None:
        store = SessionStore.create(
            root_dir=self.tmp_dir,
            workspace_root=Path('.'),
            goal='test',
            memory_run_dir=self.tmp_dir / 'mem',
        )
        store.mark_dirty()
        # _last_write_time was just set, so flush_if_dirty should skip
        result = store.flush_if_dirty()
        self.assertFalse(result)

    def test_should_force_flush(self) -> None:
        store = SessionStore.create(
            root_dir=self.tmp_dir,
            workspace_root=Path('.'),
            goal='test',
            memory_run_dir=self.tmp_dir / 'mem',
        )
        store.mark_dirty()
        store.force_flush()
        self.assertFalse(store._dirty)


class TailWindowLoadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path('tests/.tmp') / f'session-{uuid.uuid4().hex}'
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_should_load_from_session_json(self) -> None:
        store = SessionStore.create(
            root_dir=self.tmp_dir,
            workspace_root=Path('.'),
            goal='test goal',
            memory_run_dir=self.tmp_dir / 'mem',
        )
        # Append some events
        store.append_event('run_started', {'goal': 'test goal'})
        store.append_event('chat_user', {'content': 'hello', 'role': 'user'})

        # Fast load should work
        loaded = SessionStore.load_fast(
            root_dir=self.tmp_dir,
            session_id=store.state.session_id,
        )
        self.assertEqual(loaded.state.goal, 'test goal')
        self.assertEqual(loaded.state.status, 'active')

    def test_should_rebuild_state_from_events(self) -> None:
        store = SessionStore.create(
            root_dir=self.tmp_dir,
            workspace_root=Path('.'),
            goal='initial',
            memory_run_dir=self.tmp_dir / 'mem',
        )
        store.append_event('run_started', {'goal': 'updated goal'})
        store.append_event('chat_user', {'content': 'hello', 'role': 'user'})
        store.append_event('chat_assistant', {'content': 'hi there', 'role': 'assistant'})

        loaded = SessionStore.load_fast(
            root_dir=self.tmp_dir,
            session_id=store.state.session_id,
        )
        self.assertEqual(loaded.state.goal, 'updated goal')
        self.assertGreater(len(loaded.state.history_tail), 0)

    def test_should_handle_empty_events_file(self) -> None:
        store = SessionStore.create(
            root_dir=self.tmp_dir,
            workspace_root=Path('.'),
            goal='test',
            memory_run_dir=self.tmp_dir / 'mem',
        )
        # Clear events file
        store.events_file.write_text('')

        # Should fall back to session.json
        loaded = SessionStore.load_fast(
            root_dir=self.tmp_dir,
            session_id=store.state.session_id,
        )
        self.assertEqual(loaded.state.goal, 'test')

    def test_should_handle_missing_events_file(self) -> None:
        store = SessionStore.create(
            root_dir=self.tmp_dir,
            workspace_root=Path('.'),
            goal='test',
            memory_run_dir=self.tmp_dir / 'mem',
        )
        # Remove events file
        store.events_file.unlink(missing_ok=True)

        # Should fall back to session.json
        loaded = SessionStore.load_fast(
            root_dir=self.tmp_dir,
            session_id=store.state.session_id,
        )
        self.assertEqual(loaded.state.goal, 'test')

    def test_should_read_jsonl_tail_only(self) -> None:
        store = SessionStore.create(
            root_dir=self.tmp_dir,
            workspace_root=Path('.'),
            goal='test',
            memory_run_dir=self.tmp_dir / 'mem',
        )
        # Append many events to exceed tail window
        for i in range(100):
            store.append_event('chat_user', {'content': f'message {i}', 'role': 'user'})

        loaded = SessionStore.load_fast(
            root_dir=self.tmp_dir,
            session_id=store.state.session_id,
        )
        # Should have history from tail
        self.assertGreater(len(loaded.state.history_tail), 0)


if __name__ == '__main__':
    unittest.main()
