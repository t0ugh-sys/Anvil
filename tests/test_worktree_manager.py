from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401

from loop_agent.worktree_manager import WorktreeManager


class WorktreeManagerTests(unittest.TestCase):
    def test_should_create_and_cleanup_copy_workspace(self) -> None:
        root = Path('tests/.tmp') / f'worktree-{uuid.uuid4().hex}'
        source = root / 'source'
        source.mkdir(parents=True, exist_ok=True)
        (source / 'README.md').write_text('hello', encoding='utf-8')
        manager = WorktreeManager(root_dir=root / 'isolated', source_root=source, preferred_mode='copy')
        try:
            lease = manager.create('task-1')
            self.assertEqual(lease.mode, 'copy')
            self.assertTrue((lease.workspace_path / 'README.md').exists())
            self.assertNotEqual(lease.workspace_path, source)
            manager.cleanup(lease)
            self.assertFalse(lease.workspace_path.exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
