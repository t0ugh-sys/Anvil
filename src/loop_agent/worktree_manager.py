from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


IsolationMode = Literal['copy', 'git']


@dataclass(frozen=True)
class WorkspaceLease:
    task_id: str
    workspace_path: Path
    mode: IsolationMode


class WorktreeManager:
    def __init__(self, *, root_dir: Path, source_root: Path, preferred_mode: IsolationMode = 'copy') -> None:
        self.root_dir = root_dir
        self.source_root = source_root.resolve()
        self.preferred_mode = preferred_mode
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create(self, task_id: str) -> WorkspaceLease:
        if not task_id.strip():
            raise ValueError('task_id must not be empty')
        target = self.root_dir / task_id
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)

        mode = self._resolve_mode()
        if mode == 'git':
            self._create_git_worktree(target)
        else:
            self._create_copy_workspace(target)

        return WorkspaceLease(task_id=task_id, workspace_path=target, mode=mode)

    def cleanup(self, lease: WorkspaceLease) -> None:
        if lease.mode == 'git':
            self._remove_git_worktree(lease.workspace_path)
            return
        shutil.rmtree(lease.workspace_path, ignore_errors=True)

    def _resolve_mode(self) -> IsolationMode:
        if self.preferred_mode == 'git' and self._can_use_git_worktree():
            return 'git'
        return 'copy'

    def _can_use_git_worktree(self) -> bool:
        if not (self.source_root / '.git').exists():
            return False
        try:
            result = subprocess.run(
                ['git', '-C', str(self.source_root), 'rev-parse', '--is-inside-work-tree'],
                check=False,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
            )
        except Exception:
            return False
        return result.returncode == 0

    def _create_copy_workspace(self, target: Path) -> None:
        ignored_names = {'.git', '__pycache__', '.pytest_cache', '.mypy_cache', '.loopagent', 'runs'}
        try:
            relative_root = self.root_dir.resolve().relative_to(self.source_root)
            ignored_names.add(relative_root.parts[0])
        except ValueError:
            pass
        shutil.copytree(
            self.source_root,
            target,
            ignore=shutil.ignore_patterns(*sorted(ignored_names)),
        )

    def _create_git_worktree(self, target: Path) -> None:
        result = subprocess.run(
            ['git', '-C', str(self.source_root), 'worktree', 'add', '--detach', str(target), 'HEAD'],
            check=False,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
        )
        if result.returncode != 0:
            raise ValueError(f'git worktree add failed: {(result.stderr or result.stdout).strip()}')

    def _remove_git_worktree(self, target: Path) -> None:
        result = subprocess.run(
            ['git', '-C', str(self.source_root), 'worktree', 'remove', '--force', str(target)],
            check=False,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
        )
        if result.returncode != 0 and target.exists():
            raise ValueError(f'git worktree remove failed: {(result.stderr or result.stdout).strip()}')
