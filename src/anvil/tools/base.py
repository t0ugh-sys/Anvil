"""Base types and utilities for tool implementations."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Tuple

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


# ============== Input Sanitization ==============
# Based on Zero2Agent article #41: Agent Security & Protection
# Detects prompt injection patterns in user inputs and tool arguments.


# Known prompt injection patterns (case-insensitive)
_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r'ignore\s+(all\s+)?previous\s+instructions', re.IGNORECASE),
    re.compile(r'you\s+are\s+now\s+(a|an)\s+', re.IGNORECASE),
    re.compile(r'system\s*:\s*', re.IGNORECASE),
    re.compile(r'forget\s+(all\s+)?(previous|everything|prior)', re.IGNORECASE),
    re.compile(r'\[INST\].*\[/INST\]', re.IGNORECASE),
    re.compile(r'<\|im_start\|>system', re.IGNORECASE),
    re.compile(r'override\s+(all\s+)?safety', re.IGNORECASE),
    re.compile(r'do\s+not\s+(follow|obey)\s+(any\s+)?(previous|prior|earlier)', re.IGNORECASE),
    re.compile(r'disregard\s+(all\s+)?(previous|prior|earlier)', re.IGNORECASE),
    re.compile(r'new\s+instructions?\s*:', re.IGNORECASE),
    re.compile(r'act\s+as\s+if\s+you\s+(are|were)', re.IGNORECASE),
]

# Maximum input length to prevent resource exhaustion
_MAX_INPUT_LENGTH = 100_000  # 100K chars


class InjectionDetected(Exception):
    """Raised when a prompt injection attempt is detected."""

    def __init__(self, pattern: str, input_sample: str):
        self.pattern = pattern
        self.input_sample = input_sample[:200]
        super().__init__(f'Prompt injection detected: matched pattern "{pattern}"')


def detect_injection(text: str) -> List[str]:
    """Check text for prompt injection patterns.

    Returns:
        List of matched pattern descriptions. Empty list = safe.
    """
    if not text:
        return []

    matches = []
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            matches.append(pattern.pattern)
    return matches


def sanitize_input(text: str, *, max_length: int = _MAX_INPUT_LENGTH) -> str:
    """Sanitize user input for safe processing.

    - Truncates excessively long inputs
    - Strips null bytes
    - Does NOT modify content otherwise (preserves user intent)

    Use detect_injection() separately if you want to block injection attempts.
    """
    if not text:
        return ''

    # Remove null bytes
    sanitized = text.replace('\0', '')

    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    return sanitized
