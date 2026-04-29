from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

from .policies import Capability


class ToolRisk(str, Enum):
    low = 'low'
    medium = 'medium'
    high = 'high'


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    capabilities: Tuple[Capability, ...] = field(default_factory=tuple)
    risk_level: ToolRisk = ToolRisk.low
    requires_workspace: bool = True
    input_notes: str = ''

    def to_dict(self) -> dict[str, object]:
        return {
            'name': self.name,
            'description': self.description,
            'capabilities': [item.value for item in self.capabilities],
            'risk_level': self.risk_level.value,
            'requires_workspace': self.requires_workspace,
            'input_notes': self.input_notes,
        }
