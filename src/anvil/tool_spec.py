from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Tuple

from .agent_protocol import ToolResult
from .policies import Capability


class ToolRisk(str, Enum):
    low = 'low'
    medium = 'medium'
    high = 'high'


# ============== Validation ==============

@dataclass(frozen=True)
class ValidationResult:
    """Structured validation result for tool input."""
    ok: bool
    errors: Tuple[str, ...] = ()

    @classmethod
    def success(cls) -> ValidationResult:
        return cls(ok=True)

    @classmethod
    def failure(cls, *errors: str) -> ValidationResult:
        return cls(ok=False, errors=tuple(errors))

    def to_error_string(self, tool_name: str) -> str:
        if self.ok:
            return ''
        issues = '\n'.join(f'  - {e}' for e in self.errors)
        return f'{tool_name} validation failed:\n{issues}'


# ============== Tool Spec ==============

@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    capabilities: Tuple[Capability, ...] = field(default_factory=tuple)
    risk_level: ToolRisk = ToolRisk.low
    requires_workspace: bool = True
    input_notes: str = ''
    is_concurrency_safe: bool = False
    is_read_only: bool = False
    is_destructive: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            'name': self.name,
            'description': self.description,
            'capabilities': [item.value for item in self.capabilities],
            'risk_level': self.risk_level.value,
            'requires_workspace': self.requires_workspace,
            'input_notes': self.input_notes,
            'is_concurrency_safe': self.is_concurrency_safe,
            'is_read_only': self.is_read_only,
            'is_destructive': self.is_destructive,
        }


# ============== Tool Definition (build_tool factory) ==============

# Fail-closed defaults: tools are NOT concurrency-safe, NOT read-only by default.
TOOL_DEFAULTS: Dict[str, Any] = {
    'is_concurrency_safe': False,
    'is_read_only': False,
    'is_destructive': False,
}


@dataclass(frozen=True)
class ToolDef:
    """Full tool definition with handler, spec, and validation."""
    name: str
    description: str
    handler: Callable[..., ToolResult]
    capabilities: Tuple[Capability, ...] = field(default_factory=tuple)
    risk_level: ToolRisk = ToolRisk.low
    requires_workspace: bool = True
    input_notes: str = ''
    is_concurrency_safe: bool = False
    is_read_only: bool = False
    is_destructive: bool = False
    validate_input: Callable[[Dict[str, object]], ValidationResult] | None = None

    def to_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            capabilities=self.capabilities,
            risk_level=self.risk_level,
            requires_workspace=self.requires_workspace,
            input_notes=self.input_notes,
            is_concurrency_safe=self.is_concurrency_safe,
            is_read_only=self.is_read_only,
            is_destructive=self.is_destructive,
        )


def build_tool(
    name: str,
    description: str,
    handler: Callable[..., ToolResult],
    *,
    capabilities: Tuple[Capability, ...] = (),
    risk_level: ToolRisk = ToolRisk.low,
    requires_workspace: bool = True,
    input_notes: str = '',
    is_concurrency_safe: bool = False,
    is_read_only: bool = False,
    is_destructive: bool = False,
    validate_input: Callable[[Dict[str, object]], ValidationResult] | None = None,
) -> ToolDef:
    """Build a tool definition with fail-closed defaults.

    Mirrors Claude Code's buildTool() pattern: conservative defaults
    (not concurrency-safe, not read-only) that callers explicitly override.
    """
    return ToolDef(
        name=name,
        description=description,
        handler=handler,
        capabilities=capabilities,
        risk_level=risk_level,
        requires_workspace=requires_workspace,
        input_notes=input_notes,
        is_concurrency_safe=is_concurrency_safe,
        is_read_only=is_read_only,
        is_destructive=is_destructive,
        validate_input=validate_input,
    )
