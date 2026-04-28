from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .run_schema import EventRow, SCHEMA_VERSION, utc_now_iso


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


@dataclass(frozen=True)
class RunRecorder:
    run_dir: Path
    events_file: Path
    session_id: Optional[str] = None

    @classmethod
    def create(cls, base_dir: Optional[Path] = None, *, session_id: Optional[str] = None) -> 'RunRecorder':
        # Keep consistent with CLI defaults: `.loopagent/runs`
        root = (base_dir or Path('.loopagent/runs')).resolve()
        run_dir = root / _utc_timestamp()
        run_dir.mkdir(parents=True, exist_ok=True)
        events_file = run_dir / 'events.jsonl'
        return cls(run_dir=run_dir, events_file=events_file, session_id=session_id)

    def write_event(self, event: str, payload: Dict[str, Any]) -> None:
        # Stable event envelope for replay/debugging.
        step = payload.get('step')
        step_index = step if isinstance(step, int) else None
        tool_name = None
        permission_decision = None
        permission_reason = None
        metadata = payload.get('metadata', {})
        if isinstance(metadata, dict):
            tool_calls = metadata.get('tool_calls', [])
            tool_results = metadata.get('tool_results', [])
            if len(tool_calls) == 1 and len(tool_results) == 1:
                tool_call = tool_calls[0]
                tool_result = tool_results[0]
                if isinstance(tool_call, dict):
                    name = tool_call.get('name')
                    if isinstance(name, str):
                        tool_name = name
                if isinstance(tool_result, dict):
                    decision = tool_result.get('permission_decision')
                    reason = tool_result.get('permission_reason')
                    if isinstance(decision, str):
                        permission_decision = decision
                    if isinstance(reason, str):
                        permission_reason = reason
        row = EventRow(
            schema_version=SCHEMA_VERSION,
            ts=utc_now_iso(),
            event=event,
            step=step_index,
            payload=payload,
            session_id=self.session_id,
            tool_name=tool_name,
            permission_decision=permission_decision,
            permission_reason=permission_reason,
        )
        with self.events_file.open('a', encoding='utf-8') as file:
            file.write(json.dumps(row.to_dict(), ensure_ascii=False))
            file.write('\n')

    def write_summary(self, payload: Dict[str, Any]) -> None:
        summary_file = self.run_dir / 'summary.json'
        summary_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
