"""MCP (Model Context Protocol) compatible types and conversion helpers.

MCP spec: https://modelcontextprotocol.io/
Lets Anvil tools be described and called via MCP wire format so third-party
MCP clients can interact with an Anvil runtime without knowing its internals.

Conversion direction:
  Outbound (Anvil → MCP client):
    ToolDef          → MCPTool            (tool listing)
    ToolResult       → MCPCallToolResult  (tool response)

  Inbound (MCP client → Anvil):
    MCPToolCallParams → ToolCall          (tool invocation)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from ..agent.protocol import ToolCall, ToolResult

__all__ = [
    # Content types
    'MCPTextContent',
    'MCPImageContent',
    # Tool schema
    'MCPInputSchema',
    'MCPTool',
    # Request / response
    'MCPToolCallParams',
    'MCPCallToolResult',
    # Converters
    'anvil_tool_def_to_mcp',
    'anvil_tool_call_params_to_anvil',
    'mcp_result_from_anvil',
    'tool_list_to_mcp',
]


# ---------------------------------------------------------------------------
# Content types (MCP §5.3 — content blocks)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MCPTextContent:
    text: str
    type: Literal['text'] = field(default='text', init=False)

    def to_dict(self) -> Dict[str, str]:
        return {'type': self.type, 'text': self.text}


@dataclass(frozen=True)
class MCPImageContent:
    data: str           # base64-encoded
    mime_type: str
    type: Literal['image'] = field(default='image', init=False)

    def to_dict(self) -> Dict[str, str]:
        return {'type': self.type, 'data': self.data, 'mimeType': self.mime_type}


MCPContent = MCPTextContent | MCPImageContent


# ---------------------------------------------------------------------------
# Tool definition (MCP §5.1 — tools/list)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MCPInputSchema:
    """JSON Schema object describing a tool's accepted parameters.

    MCP requires at minimum ``{"type": "object"}``.  Anvil tools may carry
    richer schemas via the ``properties`` and ``required`` fields; if not
    provided the schema degrades gracefully to an open object.
    """
    properties: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    type: str = 'object'

    def to_dict(self) -> Dict[str, Any]:
        schema: Dict[str, Any] = {'type': self.type}
        if self.properties:
            schema['properties'] = self.properties
        if self.required:
            schema['required'] = self.required
        return schema


@dataclass(frozen=True)
class MCPTool:
    """MCP tool descriptor returned by ``tools/list``."""
    name: str
    description: str
    input_schema: MCPInputSchema = field(default_factory=MCPInputSchema)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'inputSchema': self.input_schema.to_dict(),
        }


# ---------------------------------------------------------------------------
# Tool call / result (MCP §5.2 — tools/call)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MCPToolCallParams:
    """Inbound MCP tool call params (the ``params`` field of tools/call)."""
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {'name': self.name, 'arguments': self.arguments}


@dataclass(frozen=True)
class MCPCallToolResult:
    """Outbound MCP tool result (return value of tools/call)."""
    content: List[MCPContent]
    is_error: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': [c.to_dict() for c in self.content],
            'isError': self.is_error,
        }


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------

def anvil_tool_def_to_mcp(tool_def: Any) -> MCPTool:
    """Convert an Anvil ``ToolDef`` to an ``MCPTool``.

    Anvil tools currently carry a plain-text ``input_notes`` rather than a
    structured JSON Schema.  This helper builds the minimal MCP schema
    (open object) and embeds ``input_notes`` in the description so MCP
    clients still get parameter documentation.
    """
    name: str = getattr(tool_def, 'name', '')
    description: str = getattr(tool_def, 'description', '')
    input_notes: str = getattr(tool_def, 'input_notes', '')

    if input_notes:
        description = f'{description}\n\nParameters:\n{input_notes}'

    return MCPTool(
        name=name,
        description=description.strip(),
        input_schema=MCPInputSchema(),
    )


def anvil_tool_call_params_to_anvil(params: MCPToolCallParams, *, call_id: str) -> ToolCall:
    """Convert an inbound MCP call to an Anvil ``ToolCall``."""
    return ToolCall(id=call_id, name=params.name, arguments=params.arguments)


def mcp_result_from_anvil(result: ToolResult) -> MCPCallToolResult:
    """Convert an Anvil ``ToolResult`` to an ``MCPCallToolResult``."""
    parts: List[MCPContent] = []
    if result.output:
        parts.append(MCPTextContent(text=result.output))
    if result.error:
        parts.append(MCPTextContent(text=result.error))
    if not parts:
        parts.append(MCPTextContent(text=''))
    return MCPCallToolResult(content=parts, is_error=not result.ok)


def tool_list_to_mcp(tool_defs: List[Any]) -> List[Dict[str, Any]]:
    """Convert a list of Anvil ``ToolDef`` objects to MCP ``tools/list`` payload."""
    return [anvil_tool_def_to_mcp(td).to_dict() for td in tool_defs]


# ---------------------------------------------------------------------------
# Parsing helpers (inbound MCP JSON → Anvil types)
# ---------------------------------------------------------------------------

def parse_mcp_tool_call(raw: Dict[str, Any], *, call_id: str) -> ToolCall:
    """Parse a raw MCP ``tools/call`` params dict to a ``ToolCall``."""
    name = raw.get('name', '')
    arguments = raw.get('arguments', {})
    if not isinstance(name, str):
        raise ValueError(f'MCP tool call name must be a string, got {type(name).__name__}')
    if not isinstance(arguments, dict):
        raise ValueError(f'MCP tool call arguments must be an object, got {type(arguments).__name__}')
    return ToolCall(id=call_id, name=name, arguments=arguments)
