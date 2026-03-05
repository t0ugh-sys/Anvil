from __future__ import annotations

"""Run artifact and event schema helpers.

This module defines a small, stable schema for run artifacts so that:
- runs can be replayed/debugged
- future changes can be versioned without breaking old runs

The schema is intentionally stdlib-only.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Final

SCHEMA_VERSION: Final[str] = "run-schema-v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class EventRow:
    schema_version: str
    ts: str
    event: str
    step: int | None
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ts": self.ts,
            "event": self.event,
            "step": self.step,
            "payload": self.payload,
        }
