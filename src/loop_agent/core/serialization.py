from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict

from .types import RunResult


def run_result_to_dict(result: RunResult[Any], *, include_history: bool = True) -> Dict[str, Any]:
    payload: Dict[str, Any] = asdict(result)
    payload['stop_reason'] = result.stop_reason.value
    state_payload = payload.get('state')
    if isinstance(state_payload, dict):
        compression_state = {
            'summary': state_payload.get('compact_summary', ''),
            'compaction_count': state_payload.get('compaction_count', 0),
            'archived_transcripts': state_payload.get('archived_transcripts', []),
            'last_compaction_reason': state_payload.get('last_compaction_reason', ''),
        }
        payload['compression_state'] = compression_state
    if not include_history:
        payload.pop('history', None)
    return payload


def run_result_to_json(result: RunResult[Any], *, include_history: bool = True) -> str:
    payload = run_result_to_dict(result, include_history=include_history)
    return json.dumps(payload, ensure_ascii=False)

