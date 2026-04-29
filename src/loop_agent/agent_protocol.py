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
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
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


def render_agent_step_schema() -> str:
    return (
        '{"thought":"...","plan":["..."],'
        '"tool_calls":[{"id":"call_1","name":"read_file","arguments":{"path":"README.md"}},'
        '{"id":"call_2","name":"apply_patch","arguments":{"patch":"*** Begin Patch\\n*** Update File: README.md\\n@@\\n-old\\n+new\\n*** End Patch"}}],'
        '"final":null}'
    )
