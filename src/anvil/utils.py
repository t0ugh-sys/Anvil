"""
Shared utility functions for Anvil.

Consolidates commonly used helpers to avoid duplication across modules.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core.types import ObserverFn


def default_run_id() -> str:
    """Generate a default run ID from current UTC timestamp."""
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def build_jsonl_observer(path: str) -> ObserverFn:
    """Create an observer that writes events as JSONL to a file."""
    def observer(event: str, payload: Dict[str, Any]) -> None:
        record = {'event': event, 'payload': payload}
        with open(path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(record, ensure_ascii=False))
            file.write('\n')

    return observer


def merge_observers(observers: List[ObserverFn]) -> Optional[ObserverFn]:
    """Merge multiple observers into one. Returns None if no observers."""
    active = [item for item in observers if item is not None]
    if not active:
        return None

    def merged(event: str, payload: Dict[str, Any]) -> None:
        for observer in active:
            observer(event, payload)

    return merged


def resolve_goal(goal: str | None = None, goal_file: str | None = None) -> str:
    """Resolve goal from direct string or file path."""
    if goal_file:
        return Path(goal_file).read_text(encoding='utf-8-sig').strip()
    return str(goal or '').strip()
