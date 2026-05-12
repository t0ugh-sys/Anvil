from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    plan = [item for item in plan_raw]

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
