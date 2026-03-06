from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, Tuple


@dataclass(frozen=True)
class MemoryContext:
    state_summary: Dict[str, Any] = field(default_factory=dict)
    last_steps: Tuple[str, ...] = field(default_factory=tuple)


class MemoryStore(Protocol):
    def on_event(self, event: str, payload: Dict[str, Any]) -> None:
        ...

    def load_context(self, *, goal: str, last_k_steps: int) -> MemoryContext:
        ...
