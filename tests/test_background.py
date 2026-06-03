"""Tests for background module — BackgroundCommandRunner and BackgroundTaskInfo."""
from __future__ import annotations

import time

import _bootstrap  # noqa: F401

from anvil.background import BackgroundCommandRunner, BackgroundTaskInfo


class TestBackgroundTaskInfo:
    def test_to_dict(self):
        info = BackgroundTaskInfo(id='bg1', command=('sleep', '1'), status='running')
        d = info.to_dict()
        assert d['id'] == 'bg1'
        assert d['command'] == ['sleep', '1']
        assert d['status'] == 'running'

    def test_frozen(self):
        info = BackgroundTaskInfo(id='bg2', command=('echo',), status='pending')
        try:
            info.id = 'changed'
            assert False, 'Should be frozen'
        except AttributeError:
            pass


class TestBackgroundCommandRunner:
    def test_init(self, tmp_path):
        runner = BackgroundCommandRunner(tmp_path)
        assert runner.workspace_root == tmp_path

    def test_snapshot_empty(self, tmp_path):
        runner = BackgroundCommandRunner(tmp_path)
        assert runner.snapshot() == ()

    def test_drain_notifications_empty(self, tmp_path):
        runner = BackgroundCommandRunner(tmp_path)
        assert runner.drain_notifications() == ()

    def test_spawn_and_wait(self, tmp_path):
        runner = BackgroundCommandRunner(tmp_path)
        result = runner.spawn(command=['echo', 'hello'], call_id='c1')
        assert result.ok
        assert 'bg_1' in result.output
        # Wait for background task to finish
        time.sleep(1.0)
        # Drain notification
        notifications = runner.drain_notifications()
        assert len(notifications) >= 1
        assert notifications[0].ok

    def test_spawn_empty_command(self, tmp_path):
        runner = BackgroundCommandRunner(tmp_path)
        result = runner.spawn(command=[], call_id='c2')
        assert not result.ok
        assert 'required' in result.error

    def test_snapshot_after_spawn(self, tmp_path):
        runner = BackgroundCommandRunner(tmp_path)
        runner.spawn(command=['echo', 'test'], call_id='c3')
        snap = runner.snapshot()
        assert len(snap) == 1
        assert snap[0].status == 'running'

    def test_kill_task_nonexistent(self, tmp_path):
        runner = BackgroundCommandRunner(tmp_path)
        assert runner.kill_task('nonexistent') is False

    def test_read_output_empty(self, tmp_path):
        runner = BackgroundCommandRunner(tmp_path)
        assert runner.read_output('nonexistent') == ''

    def test_spawn_command_not_found(self, tmp_path):
        runner = BackgroundCommandRunner(tmp_path)
        result = runner.spawn(command=['nonexistent_cmd_xyz'], call_id='c4')
        assert result.ok  # spawn itself succeeds
        time.sleep(1.0)
        notifications = runner.drain_notifications()
        assert len(notifications) >= 1
        assert not notifications[0].ok  # but task fails
