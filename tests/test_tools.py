from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401

from loop_agent.agent_protocol import ToolCall
from loop_agent.tools import (
    ToolContext,
    build_default_tools,
    builtin_tool_registrations,
    execute_tool_call,
    fetch_url_tool,
    register_tool_handler,
)


class ToolsTests(unittest.TestCase):
    def test_should_build_dispatch_from_builtin_registrations(self) -> None:
        registrations = builtin_tool_registrations()
        names = [name for name, _ in registrations]
        dispatch = build_default_tools()

        self.assertIn('read_file', names)
        self.assertIn('todo_write', names)
        self.assertIn('git_status', names)
        self.assertIn('gh_issue_list', names)
        self.assertEqual(set(names), set(dispatch.keys()))

    def test_should_register_custom_tool_handler_in_dispatch_map(self) -> None:
        def echo_tool(context: ToolContext, args):
            return type('Result', (), {})  # pragma: no cover

        dispatch_map = build_default_tools()
        updated = register_tool_handler(dispatch_map, 'echo', echo_tool)

        self.assertIs(updated['echo'], echo_tool)

    def test_should_apply_patch_update_file(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'tools-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        target = tmp_dir / 'README.md'
        target.write_text('hello\nworld\n', encoding='utf-8')
        try:
            patch = '\n'.join(
                [
                    '*** Begin Patch',
                    '*** Update File: README.md',
                    '@@',
                    '-hello',
                    '+hi',
                    '*** End Patch',
                ]
            )
            call = ToolCall(id='call_1', name='apply_patch', arguments={'patch': patch})
            result = execute_tool_call(ToolContext(workspace_root=tmp_dir), call, build_default_tools())
            self.assertTrue(result.ok, msg=result.error or '')
            self.assertIn('README.md', result.output)
            self.assertEqual(target.read_text(encoding='utf-8'), 'hi\nworld\n')
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_reject_patch_outside_workspace(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'tools-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            patch = '\n'.join(
                [
                    '*** Begin Patch',
                    '*** Add File: ../escape.txt',
                    '+x',
                    '*** End Patch',
                ]
            )
            call = ToolCall(id='call_1', name='apply_patch', arguments={'patch': patch})
            result = execute_tool_call(ToolContext(workspace_root=tmp_dir), call, build_default_tools())
            self.assertFalse(result.ok)
            self.assertIn('workspace', result.error or '')
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_skip_ignored_directories_during_search(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'tools-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            (tmp_dir / '.git').mkdir()
            (tmp_dir / '.git' / 'config').write_text('needle', encoding='utf-8')
            (tmp_dir / 'docs').mkdir()
            (tmp_dir / 'docs' / 'note.txt').write_text('needle', encoding='utf-8')

            call = ToolCall(id='call_1', name='search', arguments={'pattern': 'needle'})
            result = execute_tool_call(ToolContext(workspace_root=tmp_dir), call, build_default_tools())

            self.assertTrue(result.ok, msg=result.error or '')
            self.assertEqual(result.output, 'docs/note.txt')
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_strip_script_and_style_when_fetching_url(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return (
                    b'<html><head><style>.x{color:red}</style></head>'
                    b'<body><script>alert(1)</script><h1>Hello</h1><p>World</p></body></html>'
                )

        with patch('urllib.request.urlopen', return_value=FakeResponse()):
            result = fetch_url_tool(ToolContext(workspace_root=Path('.')), {'id': 'call_1', 'url': 'https://example.com'})

        self.assertTrue(result.ok, msg=result.error or '')
        self.assertIn('Hello', result.output)
        self.assertIn('World', result.output)
        self.assertNotIn('alert(1)', result.output)
        self.assertNotIn('color:red', result.output)


if __name__ == '__main__':
    unittest.main()
