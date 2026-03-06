from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class JsonDecision:
    answer: str
    done: bool
    raw: Dict[str, Any]


def parse_json_decision(text: str) -> Optional[JsonDecision ]:
    candidate = text.strip()
    if candidate.startswith('```'):
        lines = [line for line in candidate.splitlines() if line.strip()]
        if len(lines) >= 3 and lines[0].startswith('```') and lines[-1].startswith('```'):
            candidate = '\n'.join(lines[1:-1]).strip()

    try:
        raw = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if not isinstance(raw, dict):
        return None
    answer = raw.get('answer')
    done = raw.get('done')
    if not isinstance(answer, str) or not isinstance(done, bool):
        return None
    return JsonDecision(answer=answer, done=done, raw=raw)

