"""Async variants of team runtime tests — mirrors test_team_runtime.py."""
from __future__ import annotations

import asyncio
import shutil
import unittest
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401

from anvil.runtime.team import (
    PersistentTeamRuntime,
    PersistentTeammateSpec,
    TeamMessageType,
)
from anvil.runtime.task_graph import Task, TaskStatus


def _build_mock_decider(prefix: str):
    def decider(goal, history, tool_results, state_summary, last_steps) -> str:
        return (
            '{"thought":"'
            + prefix
            + '","plan":[],"tool_calls":[],"final":"'
            + prefix
            + ': '
            + goal.replace('"', "'")
            + '"}'
        )
    return decider


class AsyncTeamRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_spawn_async_persists_config(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'ateam-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        runtime = PersistentTeamRuntime(tmp_dir / '.team')
        try:
            await runtime.spawn_teammate_async(
                PersistentTeammateSpec(
                    name='alice',
                    role='coder',
                    workspace_root=tmp_dir,
                    decider=_build_mock_decider('alice'),
                )
            )
            await asyncio.sleep(0.1)
            config_path = tmp_dir / '.team' / 'config.json'
            self.assertTrue(config_path.exists())
            self.assertIn('"name": "alice"', config_path.read_text(encoding='utf-8'))
        finally:
            await runtime.shutdown_all_async()
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def test_async_deliver_message_and_receive_reply(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'ateam-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        runtime = PersistentTeamRuntime(tmp_dir / '.team')
        try:
            await runtime.spawn_teammate_async(
                PersistentTeammateSpec(
                    name='alice',
                    role='coder',
                    workspace_root=tmp_dir,
                    decider=_build_mock_decider('alice'),
                )
            )
            runtime.send_message('alice', 'fix bug', sender='lead')

            for _ in range(40):
                await asyncio.sleep(0.05)
                inbox = runtime.inbox_store.drain('lead')
                if inbox:
                    self.assertEqual(inbox[0].sender, 'alice')
                    self.assertEqual(inbox[0].message_type, TeamMessageType.message)
                    self.assertIn('alice: fix bug', inbox[0].body)
                    return
            self.fail('expected alice reply in lead inbox')
        finally:
            await runtime.shutdown_all_async()
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def test_async_broadcast_to_all_teammates(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'ateam-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        runtime = PersistentTeamRuntime(tmp_dir / '.team')
        try:
            await runtime.spawn_teammate_async(
                PersistentTeammateSpec(
                    name='alice',
                    role='coder',
                    workspace_root=tmp_dir,
                    decider=_build_mock_decider('alice'),
                )
            )
            await runtime.spawn_teammate_async(
                PersistentTeammateSpec(
                    name='bob',
                    role='reviewer',
                    workspace_root=tmp_dir,
                    decider=_build_mock_decider('bob'),
                )
            )
            runtime.broadcast('inspect patch', sender='lead')

            seen = set()
            for _ in range(60):
                await asyncio.sleep(0.05)
                inbox = runtime.inbox_store.drain('lead')
                for item in inbox:
                    seen.add(item.sender)
                if seen == {'alice', 'bob'}:
                    return
            self.fail(f'expected both teammates to respond, got {seen}')
        finally:
            await runtime.shutdown_all_async()
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def test_async_shutdown_teammate_gracefully(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'ateam-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        runtime = PersistentTeamRuntime(tmp_dir / '.team')
        try:
            await runtime.spawn_teammate_async(
                PersistentTeammateSpec(
                    name='alice',
                    role='coder',
                    workspace_root=tmp_dir,
                    decider=_build_mock_decider('alice'),
                )
            )
            runtime.shutdown_teammate('alice', sender='lead')

            for _ in range(40):
                await asyncio.sleep(0.05)
                inbox = runtime.inbox_store.drain('lead')
                if inbox:
                    self.assertEqual(inbox[0].message_type, TeamMessageType.shutdown_response)
                    self.assertEqual(runtime.teammate_status('alice'), 'shutdown')
                    return
            self.fail('expected shutdown response from alice')
        finally:
            await runtime.shutdown_all_async()
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def test_async_dispatch_ready_tasks(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'ateam-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        runtime = PersistentTeamRuntime(tmp_dir / '.team')
        try:
            await runtime.spawn_teammate_async(
                PersistentTeammateSpec(
                    name='alice',
                    role='coder',
                    workspace_root=tmp_dir,
                    decider=_build_mock_decider('alice'),
                )
            )
            await runtime.spawn_teammate_async(
                PersistentTeammateSpec(
                    name='bob',
                    role='reviewer',
                    workspace_root=tmp_dir,
                    decider=_build_mock_decider('bob'),
                )
            )
            runtime.replace_task_graph([
                Task(id='t1', title='Inspect', goal='inspect README', metadata={'role': 'coder'}),
                Task(id='t2', title='Review', goal='review README', dependencies=('t1',), metadata={'role': 'reviewer'}),
            ])

            assigned = await runtime.dispatch_ready_tasks_async(sender='lead')
            self.assertEqual(assigned, ('t1',))

            for _ in range(40):
                await asyncio.sleep(0.05)
                graph = runtime.load_task_graph()
                if graph.get_task('t2').status == TaskStatus.completed:
                    break
            final_graph = runtime.load_task_graph()
            self.assertEqual(final_graph.get_task('t1').status, TaskStatus.completed)
            self.assertEqual(final_graph.get_task('t2').status, TaskStatus.completed)
        finally:
            await runtime.shutdown_all_async()
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
