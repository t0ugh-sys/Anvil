from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    'ToolCall',
    'ToolResult',
    'AgentStep',
    'render_agent_step_schema',
    'parse_agent_step',
]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    id: str
    ok: bool
    output: str
    error: Optional[str ] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentStep:
    thought: str
    plan: List[str] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    final: Optional[str ] = None

    @property
    def done(self) -> bool:
        return self.final is not None


def parse_agent_step(raw: str) -> Optional[AgentStep ]:
    # Strip markdown code blocks if present
    text = raw.strip()
    if text.startswith('```'):
        # Remove ```json or ``` at the start
        first_newline = text.find('\n')
        if first_newline != -1:
            text = text[first_newline + 1:]
        # Remove ``` at the end
        if text.endswith('```'):
            text = text[:-3].rstrip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = _load_first_json_object(text)
    if not isinstance(payload, dict):
        return None

    thought = payload.get('thought', '')
    if not isinstance(thought, str):
        return None

    plan_raw = payload.get('plan', [])
    if not isinstance(plan_raw, list) or any(not isinstance(item, str) for item in plan_raw):
        return None
    plan = list(plan_raw)

    final = payload.get('final')
    if final is not None and not isinstance(final, str):
        return None

    tool_calls_raw = payload.get('tool_calls', [])
    if not isinstance(tool_calls_raw, list):
        return None
    tool_calls: List[ToolCall] = []
    for item in tool_calls_raw:
        if not isinstance(item, dict):
            return None
        call_id = item.get('id')
        name = item.get('name')
        arguments = item.get('arguments', {})
        if not isinstance(call_id, str) or not isinstance(name, str) or not isinstance(arguments, dict):
            return None
        tool_calls.append(ToolCall(id=call_id, name=name, arguments=arguments))

    return AgentStep(thought=thought, plan=plan, tool_calls=tool_calls, final=final)


def _load_first_json_object(text: str) -> Any:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != '{':
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    # Fallback: try regex extraction + repair for common LLM output issues
    return _repair_and_parse_json(text)


def _repair_and_parse_json(text: str) -> Any:
    """Attempt to extract and repair malformed JSON from LLM output.

    Handles common issues: trailing commas, single quotes, missing
    closing brackets, and text wrapping around JSON.
    """
    import re

    # Strategy 1: Extract {...} block via brace matching
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                result = _try_parse_repaired(candidate)
                if result is not None:
                    return result
                break

    # Strategy 2: Greedy regex for last-ditch extraction
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        result = _try_parse_repaired(match.group())
        if result is not None:
            return result

    return None


def _try_parse_repaired(text: str) -> Any:
    """Try parsing JSON with progressive repair strategies."""
    # Attempt 1: raw parse
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    # Attempt 2: remove trailing commas before } or ]
    import re
    repaired = re.sub(r',\s*([}\]])', r'\1', text)
    try:
        obj = json.loads(repaired)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    # Attempt 3: replace single quotes with double quotes (crude but effective)
    repaired = text.replace("'", '"')
    try:
        obj = json.loads(repaired)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    return None


def render_agent_step_schema() -> str:
    return (
        'Required JSON object fields: '
        'thought:string, '
        'plan:string[], '
        'tool_calls:{id:string,name:string,arguments:object}[], '
        'final:string|null. '
        'Use final only after the requested work is complete.'
    )
