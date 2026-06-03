"""Base types and utilities for tool implementations."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Iterable, Tuple

from ..agent_protocol import ToolResult
from ..background import BackgroundCommandRunner
from ..compression import CompactManager
from ..policies import ToolPolicy
from ..tool_spec import ToolDef, ToolRisk, ToolSpec, ValidationResult

if TYPE_CHECKING:
    from ..skills import SkillLoader
    from ..todo import TodoManager

__all__ = [
    'ToolContext',
    'ToolDispatchMap',
    'ToolFn',
    'ToolRegistration',
    'require_params',
    'iter_searchable_files',
    'resolve_inside_workspace',
]


def require_params(*names: str) -> Callable[[Dict[str, object]], ValidationResult]:
    """Create a validator that checks for required parameters."""
    def validate(args: Dict[str, object]) -> ValidationResult:
        missing = [n for n in names if not args.get(n)]
        if missing:
            return ValidationResult.failure(
                *(f'required parameter `{n}` is missing' for n in missing)
            )
        return ValidationResult.success()
    return validate


@dataclass(frozen=True)
class ToolContext:
    """Context passed to all tool implementations."""
    workspace_root: Path
    policy: ToolPolicy = ToolPolicy.allow_all()
    todo_manager: 'TodoManager | None' = None
    skill_loader: 'SkillLoader | None' = None
    compact_manager: CompactManager | None = None
    background_runner: BackgroundCommandRunner | None = None
    dry_run: bool = False  # When True, write tools return preview without executing


ToolFn = Callable[[ToolContext, Dict[str, object]], ToolResult]
ToolDispatchMap = Dict[str, ToolFn]
ToolRegistration = Tuple[str, ToolFn]


# Skip directories for search operations
SEARCH_SKIP_DIRS = frozenset({
    '.git',
    '.anvil',
    '.mypy_cache',
    '.pytest_cache',
    '.ruff_cache',
    '.venv',
    '__pycache__',
    'build',
    'dist',
    'node_modules',
})


def iter_searchable_files(workspace_root: Path) -> Iterable[Path]:
    """Iterate over files in workspace, skipping common non-searchable dirs."""
    for current_root, dir_names, file_names in os.walk(workspace_root):
        dir_names[:] = [name for name in dir_names if name not in SEARCH_SKIP_DIRS]
        root_path = Path(current_root)
        for file_name in file_names:
            yield root_path / file_name


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Polyfill for Path.is_relative_to (Python 3.9+)."""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_inside_workspace(workspace_root: Path, relative_path: str) -> Path:
    """Resolve a relative path, ensuring it stays within workspace."""
    if '\0' in relative_path:
        raise ValueError('path contains null bytes')
    if any(part == '..' for part in relative_path.replace('\\', '/').split('/')):
        raise ValueError('path escapes workspace root')
    root = workspace_root.resolve()
    target = (workspace_root / relative_path).resolve()
    if not _is_relative_to(target, root):
        raise ValueError('path escapes workspace root')
    return target
