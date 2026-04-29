from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .run_schema import utc_now_iso


@dataclass(frozen=True)
class SessionEvent:
    event_type: str
    payload: dict[str, Any]
    timestamp: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'type': self.event_type,
            'timestamp': self.timestamp or utc_now_iso(),
            'payload': dict(self.payload),
        }
