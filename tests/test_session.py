from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401

from anvil.session import SessionStore


class SessionStoreTests(unittest.TestCase):
    def test_should_create_session_files(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'session-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            store = SessionStore.create(
                root_dir=tmp_dir,
                workspace_root=tmp_dir,
                goal='inspect repo',
                memory_run_dir=tmp_dir / 'memory' / 'r1',
            )
            self.assertTrue((store.session_dir / 'session.json').exists())
            self.assertTrue((store.session_dir / 'summary.json').exists())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_restore_goal_history_and_permission_cache(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'session-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            store = SessionStore.create(
                root_dir=tmp_dir,
                workspace_root=tmp_dir,
                goal='inspect repo',
                memory_run_dir=tmp_dir / 'memory' / 'r1',
                session_id='sess-1',
            )
            store.record_permission_cache({'write_file:write': 'deny'})
            store.append_event('chat_command', {'command': 'status', 'argument': ''})
            store.append_event('chat_user', {'role': 'user', 'content': 'inspect runtime'})
            store.append_event(
                'step_succeeded',
                {
                    'step': 0,
                    'output': 'continue',
                    'metadata': {
                        'todo_state': {'items': [], 'lines': [], 'rounds_since_update': 0},
                        'tool_calls': [{'id': 'call_1', 'name': 'write_file', 'arguments': {'path': 'x.txt'}}],
                        'tool_results': [
                            {
                                'id': 'call_1',
                                'ok': False,
                                'error': 'blocked',
                                'output': '',
                                'permission_decision': 'deny',
                                'permission_reason': 'blocked',
                            }
                        ],
                        'compression_state': {'summary': 'summary text'},
                    },
                },
            )
            restored = SessionStore.load(root_dir=tmp_dir, session_id='sess-1')
            self.assertEqual(restored.state.goal, 'inspect repo')
            self.assertEqual(restored.state.history_tail[-1], 'continue')
            self.assertEqual(restored.state.permission_cache['write_file:write'], 'deny')
            self.assertEqual(restored.state.last_summary, 'summary text')
            self.assertEqual(restored.state.command_count, 1)
            self.assertEqual(restored.state.turn_count, 1)
            self.assertEqual(restored.state.message_count, 1)
            self.assertEqual(restored.state.step_count, 1)
            self.assertTrue(restored.state.last_activity_at)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_append_events_and_write_summary(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'session-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            store = SessionStore.create(
                root_dir=tmp_dir,
                workspace_root=tmp_dir,
                goal='inspect repo',
                memory_run_dir=tmp_dir / 'memory' / 'r1',
            )
            store.append_event('run_started', {'goal': 'inspect repo'})
            store.write_summary({'stop_reason': 'done', 'steps': 2})
            events_text = (store.session_dir / 'events.jsonl').read_text(encoding='utf-8')
            summary = json.loads((store.session_dir / 'summary.json').read_text(encoding='utf-8'))
            self.assertIn('"event": "run_started"', events_text)
            self.assertIn('permission_stats', summary)
            self.assertEqual(summary['stop_reason'], 'done')
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
