from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


@dataclass(frozen=True)
class RunRecorder:
    run_dir: Path
    events_file: Path

    @classmethod
    def create(cls, base_dir: Path | None = None) -> 'RunRecorder':
        root = (base_dir or Path('runs')).resolve()
        run_dir = root / _utc_timestamp()
        run_dir.mkdir(parents=True, exist_ok=True)
        events_file = run_dir / 'events.jsonl'
        return cls(run_dir=run_dir, events_file=events_file)

    def write_event(self, event: str, payload: dict[str, Any]) -> None:
        row = {'event': event, 'payload': payload}
        with self.events_file.open('a', encoding='utf-8') as file:
            file.write(json.dumps(row, ensure_ascii=False))
            file.write('\n')

    def write_summary(self, payload: dict[str, Any]) -> None:
        summary_file = self.run_dir / 'summary.json'
        summary_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
