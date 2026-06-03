"""Tests for todo module — TodoManager, TodoItem, TodoSnapshot."""
from __future__ import annotations

import _bootstrap  # noqa: F401

from anvil.todo import TodoItem, TodoManager, TodoSnapshot, render_todo_lines


class TestTodoItem:
    def test_to_dict(self):
        item = TodoItem(id='1', content='Fix bug', status='in_progress')
        d = item.to_dict()
        assert d == {'id': '1', 'content': 'Fix bug', 'status': 'in_progress'}

    def test_default_status(self):
        item = TodoItem(id='1', content='Test')
        assert item.status == 'pending'


class TestTodoManager:
    def test_write_valid(self):
        mgr = TodoManager()
        items = [
            {'id': '1', 'content': 'Task A', 'status': 'pending'},
            {'id': '2', 'content': 'Task B', 'status': 'in_progress'},
        ]
        result = mgr.write(items)
        assert len(result) == 2
        assert result[0].id == '1'
        assert result[1].status == 'in_progress'

    def test_write_rejects_duplicate_ids(self):
        mgr = TodoManager()
        try:
            mgr.write([
                {'id': '1', 'content': 'A'},
                {'id': '1', 'content': 'B'},
            ])
            assert False, 'Should raise'
        except ValueError as e:
            assert 'duplicate' in str(e)

    def test_write_rejects_multiple_in_progress(self):
        mgr = TodoManager()
        try:
            mgr.write([
                {'id': '1', 'content': 'A', 'status': 'in_progress'},
                {'id': '2', 'content': 'B', 'status': 'in_progress'},
            ])
            assert False, 'Should raise'
        except ValueError as e:
            assert 'one' in str(e).lower()

    def test_write_rejects_empty_content(self):
        mgr = TodoManager()
        try:
            mgr.write([{'id': '1', 'content': ''}])
            assert False, 'Should raise'
        except ValueError as e:
            assert 'content' in str(e).lower()

    def test_write_rejects_invalid_status(self):
        mgr = TodoManager()
        try:
            mgr.write([{'id': '1', 'content': 'A', 'status': 'invalid'}])
            assert False, 'Should raise'
        except ValueError as e:
            assert 'status' in str(e).lower()

    def test_snapshot_tracks_update(self):
        mgr = TodoManager()
        mgr.write([{'id': '1', 'content': 'A'}])
        snap = mgr.snapshot(previous_rounds_since_update=5)
        assert snap.rounds_since_update == 0  # was updated

    def test_snapshot_tracks_no_update(self):
        mgr = TodoManager()
        snap = mgr.snapshot(previous_rounds_since_update=3)
        assert snap.rounds_since_update == 4  # incremented


class TestRenderTodoLines:
    def test_empty(self):
        assert render_todo_lines([]) == []

    def test_renders_markers(self):
        items = [
            TodoItem(id='1', content='Task A', status='pending'),
            TodoItem(id='2', content='Task B', status='in_progress'),
            TodoItem(id='3', content='Task C', status='completed'),
        ]
        lines = render_todo_lines(items)
        assert len(lines) == 3
        assert '[ ]' in lines[0]
        assert '[>]' in lines[1]
        assert '[x]' in lines[2]


class TestTodoSnapshot:
    def test_defaults(self):
        snap = TodoSnapshot()
        assert snap.items == ()
        assert snap.rounds_since_update == 0
