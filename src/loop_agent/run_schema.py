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
    session_id: Optional[str] = None
    tool_name: Optional[str] = None
    permission_decision: Optional[str] = None
    permission_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "ts": self.ts,
            "event": self.event,
            "step": self.step,
            "payload": self.payload,
        }
        if self.session_id is not None:
            payload["session_id"] = self.session_id
        if self.tool_name is not None:
            payload["tool_name"] = self.tool_name
        if self.permission_decision is not None:
            payload["permission_decision"] = self.permission_decision
        if self.permission_reason is not None:
            payload["permission_reason"] = self.permission_reason
        return payload
