"""Run artifact and event schema helpers.

This module defines a small, stable schema for run artifacts so that:
- runs can be replayed/debugged
- future changes can be versioned without breaking old runs

The schema is intentionally stdlib-only.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Final, Optional

SCHEMA_VERSION: Final[str] = "run-schema-v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class EventRow:
    schema_version: str
    ts: str
    event: str
    step: Optional[int]
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ts": self.ts,
            "event": self.event,
            "step": self.step,
            "payload": self.payload,
        }
