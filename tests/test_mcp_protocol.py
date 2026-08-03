"""Tests for MCP protocol types and converters."""
from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

from anvil.agent.protocol import ToolCall, ToolResult
from anvil.protocols.mcp import (
    MCPCallToolResult,
    MCPImageContent,
    MCPInputSchema,
    MCPTextContent,
    MCPTool,
    MCPToolCallParams,
    anvil_tool_call_params_to_anvil,
    anvil_tool_def_to_mcp,
    mcp_result_from_anvil,
    parse_mcp_tool_call,
    tool_list_to_mcp,
)


# ---------------------------------------------------------------------------
# Minimal ToolDef stand-in (avoids importing heavy ToolDef dependencies)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _FakeToolDef:
    name: str
    description: str
    input_notes: str = ''


# ---------------------------------------------------------------------------
# MCPTextContent / MCPImageContent
# ---------------------------------------------------------------------------

class TestMCPContent(unittest.TestCase):

    def test_text_content_to_dict(self):
        c = MCPTextContent(text='hello')
        d = c.to_dict()
        self.assertEqual(d, {'type': 'text', 'text': 'hello'})

    def test_image_content_to_dict(self):
        c = MCPImageContent(data='abc123', mime_type='image/png')
        d = c.to_dict()
        self.assertEqual(d, {'type': 'image', 'data': 'abc123', 'mimeType': 'image/png'})

    def test_text_content_type_field_is_fixed(self):
        c = MCPTextContent(text='x')
        self.assertEqual(c.type, 'text')

    def test_image_content_type_field_is_fixed(self):
        c = MCPImageContent(data='', mime_type='image/jpeg')
        self.assertEqual(c.type, 'image')


# ---------------------------------------------------------------------------
# MCPInputSchema
# ---------------------------------------------------------------------------

class TestMCPInputSchema(unittest.TestCase):

    def test_empty_schema(self):
        s = MCPInputSchema()
        self.assertEqual(s.to_dict(), {'type': 'object'})

    def test_schema_with_properties(self):
        s = MCPInputSchema(
            properties={'path': {'type': 'string', 'description': 'file path'}},
            required=['path'],
        )
        d = s.to_dict()
        self.assertEqual(d['type'], 'object')
        self.assertIn('properties', d)
        self.assertIn('required', d)
        self.assertEqual(d['required'], ['path'])

    def test_schema_omits_empty_properties(self):
        s = MCPInputSchema(properties={}, required=[])
        d = s.to_dict()
        self.assertNotIn('properties', d)
        self.assertNotIn('required', d)


# ---------------------------------------------------------------------------
# MCPTool
# ---------------------------------------------------------------------------

class TestMCPTool(unittest.TestCase):

    def test_to_dict_minimal(self):
        tool = MCPTool(name='read_file', description='Read a file')
        d = tool.to_dict()
        self.assertEqual(d['name'], 'read_file')
        self.assertEqual(d['description'], 'Read a file')
        self.assertIn('inputSchema', d)
        self.assertEqual(d['inputSchema']['type'], 'object')

    def test_to_dict_with_schema(self):
        schema = MCPInputSchema(
            properties={'path': {'type': 'string'}},
            required=['path'],
        )
        tool = MCPTool(name='read_file', description='Read', input_schema=schema)
        d = tool.to_dict()
        self.assertEqual(d['inputSchema']['properties'], {'path': {'type': 'string'}})


# ---------------------------------------------------------------------------
# MCPCallToolResult
# ---------------------------------------------------------------------------

class TestMCPCallToolResult(unittest.TestCase):

    def test_success_result(self):
        r = MCPCallToolResult(content=[MCPTextContent(text='done')], is_error=False)
        d = r.to_dict()
        self.assertFalse(d['isError'])
        self.assertEqual(d['content'], [{'type': 'text', 'text': 'done'}])

    def test_error_result(self):
        r = MCPCallToolResult(content=[MCPTextContent(text='oops')], is_error=True)
        d = r.to_dict()
        self.assertTrue(d['isError'])

    def test_multiple_content_blocks(self):
        r = MCPCallToolResult(content=[
            MCPTextContent(text='output'),
            MCPTextContent(text='error details'),
        ])
        self.assertEqual(len(r.to_dict()['content']), 2)


# ---------------------------------------------------------------------------
# Converters — anvil_tool_def_to_mcp
# ---------------------------------------------------------------------------

class TestAnvilToolDefToMcp(unittest.TestCase):

    def test_basic_conversion(self):
        td = _FakeToolDef(name='shell', description='Run a shell command')
        mcp = anvil_tool_def_to_mcp(td)
        self.assertEqual(mcp.name, 'shell')
        self.assertIn('Run a shell command', mcp.description)

    def test_input_notes_appended_to_description(self):
        td = _FakeToolDef(name='shell', description='Run cmd', input_notes='command: str')
        mcp = anvil_tool_def_to_mcp(td)
        self.assertIn('Parameters:', mcp.description)
        self.assertIn('command: str', mcp.description)

    def test_no_input_notes_no_parameters_section(self):
        td = _FakeToolDef(name='shell', description='Run cmd', input_notes='')
        mcp = anvil_tool_def_to_mcp(td)
        self.assertNotIn('Parameters:', mcp.description)

    def test_input_schema_is_open_object(self):
        td = _FakeToolDef(name='x', description='y')
        mcp = anvil_tool_def_to_mcp(td)
        self.assertEqual(mcp.input_schema.type, 'object')
        self.assertEqual(mcp.input_schema.properties, {})


# ---------------------------------------------------------------------------
# Converters — anvil_tool_call_params_to_anvil
# ---------------------------------------------------------------------------

class TestAnvilToolCallParamsToAnvil(unittest.TestCase):

    def test_roundtrip(self):
        params = MCPToolCallParams(name='read_file', arguments={'path': 'README.md'})
        call = anvil_tool_call_params_to_anvil(params, call_id='c1')
        self.assertEqual(call.id, 'c1')
        self.assertEqual(call.name, 'read_file')
        self.assertEqual(call.arguments, {'path': 'README.md'})

    def test_empty_arguments_default(self):
        params = MCPToolCallParams(name='list_files')
        call = anvil_tool_call_params_to_anvil(params, call_id='c2')
        self.assertEqual(call.arguments, {})


# ---------------------------------------------------------------------------
# Converters — mcp_result_from_anvil
# ---------------------------------------------------------------------------

class TestMcpResultFromAnvil(unittest.TestCase):

    def test_ok_result(self):
        r = ToolResult(id='t1', ok=True, output='file contents')
        mcp = mcp_result_from_anvil(r)
        self.assertFalse(mcp.is_error)
        texts = [c.text for c in mcp.content if isinstance(c, MCPTextContent)]
        self.assertIn('file contents', texts)

    def test_error_result(self):
        r = ToolResult(id='t2', ok=False, output='', error='not found')
        mcp = mcp_result_from_anvil(r)
        self.assertTrue(mcp.is_error)
        texts = [c.text for c in mcp.content if isinstance(c, MCPTextContent)]
        self.assertIn('not found', texts)

    def test_output_and_error_both_included(self):
        r = ToolResult(id='t3', ok=False, output='partial', error='some error')
        mcp = mcp_result_from_anvil(r)
        texts = [c.text for c in mcp.content if isinstance(c, MCPTextContent)]
        self.assertIn('partial', texts)
        self.assertIn('some error', texts)

    def test_empty_result_has_one_content_block(self):
        r = ToolResult(id='t4', ok=True, output='')
        mcp = mcp_result_from_anvil(r)
        self.assertEqual(len(mcp.content), 1)
        self.assertEqual(mcp.content[0].text, '')


# ---------------------------------------------------------------------------
# Converters — parse_mcp_tool_call
# ---------------------------------------------------------------------------

class TestParseMcpToolCall(unittest.TestCase):

    def test_valid_call(self):
        raw = {'name': 'read_file', 'arguments': {'path': 'x.py'}}
        call = parse_mcp_tool_call(raw, call_id='c1')
        self.assertEqual(call.name, 'read_file')
        self.assertEqual(call.arguments, {'path': 'x.py'})

    def test_missing_arguments_defaults_to_empty(self):
        raw = {'name': 'list_files'}
        call = parse_mcp_tool_call(raw, call_id='c2')
        self.assertEqual(call.arguments, {})

    def test_invalid_name_type_raises(self):
        with self.assertRaises(ValueError):
            parse_mcp_tool_call({'name': 123, 'arguments': {}}, call_id='c3')

    def test_invalid_arguments_type_raises(self):
        with self.assertRaises(ValueError):
            parse_mcp_tool_call({'name': 'x', 'arguments': 'not a dict'}, call_id='c4')


# ---------------------------------------------------------------------------
# tool_list_to_mcp
# ---------------------------------------------------------------------------

class TestToolListToMcp(unittest.TestCase):

    def test_empty_list(self):
        self.assertEqual(tool_list_to_mcp([]), [])

    def test_single_tool(self):
        td = _FakeToolDef(name='search', description='Search files')
        result = tool_list_to_mcp([td])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['name'], 'search')
        self.assertIn('inputSchema', result[0])

    def test_multiple_tools(self):
        tools = [
            _FakeToolDef(name='read', description='Read'),
            _FakeToolDef(name='write', description='Write'),
        ]
        result = tool_list_to_mcp(tools)
        names = [r['name'] for r in result]
        self.assertIn('read', names)
        self.assertIn('write', names)


if __name__ == '__main__':
    unittest.main()
