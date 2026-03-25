from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Mapping, Tuple


class Capability(str, Enum):
    read = 'read'
    write = 'write'
    execute = 'execute'
    network = 'network'
    memory = 'memory'


TOOL_CAPABILITIES: Dict[str, Tuple[Capability, ...]] = {
    'read_file': (Capability.read,),
    'search': (Capability.read,),
    'write_file': (Capability.write,),
    'apply_patch': (Capability.write,),
    'run_command': (Capability.execute,),
    'web_search': (Capability.network,),
    'fetch_url': (Capability.network,),
    'analyze_memory': (Capability.memory, Capability.read),
}


@dataclass(frozen=True)
class ToolPolicy:
    allowed: Tuple[Capability, ...] = field(default_factory=tuple)
    denied: Tuple[Capability, ...] = field(default_factory=tuple)

    @classmethod
    def allow_all(cls) -> 'ToolPolicy':
        return cls(allowed=tuple(Capability))

    @classmethod
    def read_only(cls) -> 'ToolPolicy':
        return cls(allowed=(Capability.read, Capability.memory))

    def allows_tool(self, tool_name: str) -> bool:
        required = TOOL_CAPABILITIES.get(tool_name, tuple())
        if not required:
            return True
        denied = set(self.denied)
        allowed = set(self.allowed)
        if any(capability in denied for capability in required):
            return False
        return all(capability in allowed for capability in required)

    def denied_capabilities_for_tool(self, tool_name: str) -> Tuple[Capability, ...]:
        required = TOOL_CAPABILITIES.get(tool_name, tuple())
        denied = set(self.denied)
        allowed = set(self.allowed)
        blocked = [capability for capability in required if capability in denied or capability not in allowed]
        return tuple(blocked)

    def to_dict(self) -> Dict[str, object]:
        return {
            'allowed': [capability.value for capability in self.allowed],
            'denied': [capability.value for capability in self.denied],
        }


def policy_from_name(name: str) -> ToolPolicy:
    normalized = name.strip().lower()
    if normalized in {'full', 'allow_all'}:
        return ToolPolicy.allow_all()
    if normalized in {'read_only', 'readonly'}:
        return ToolPolicy.read_only()
    raise ValueError(f'unknown policy preset: {name}')


def build_tool_permissions() -> Mapping[str, Tuple[Capability, ...]]:
    return dict(TOOL_CAPABILITIES)
