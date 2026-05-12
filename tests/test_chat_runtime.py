from __future__ import annotations

import io
import shutil
import time
import unittest
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401

from anvil.services.chat_runtime import InteractiveRuntime
from anvil.session import SessionStore
from anvil.tools import builtin_tool_specs
from anvil.ui.chrome import ASSISTANT, BORDER, PROMPT, PROMPT_MARKER, RESPONSE_MARKER, TOP_LEFT


class TtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


class Utf8BytesInput:
    def __init__(self, value: str) -> None:
        self.buffer = io.BytesIO(value.encode('utf-8'))

    def readline(self) -> str:
        return self.buffer.readline().decode('utf-8')


class ChatRuntimeTests(unittest.TestCase):
    def test_should_render_claude_code_style_interactive_shell(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'chat-runtime-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            session_store = SessionStore.create(
                root_dir=tmp_dir / 'sessions',
                workspace_root=tmp_dir,
                goal='',
                memory_run_dir=tmp_dir / 'runs',
                session_id='sess-ui',
            )
            stdin = io.StringIO('hello\n/model\n/model other-model\n?\n/exit\n')
            stdout = io.StringIO()
            runtime = InteractiveRuntime(
                session_store=session_store,
                tool_specs=builtin_tool_specs(),
                run_turn=lambda text: f'done: {text}',
                stdin=stdin,
                stdout=stdout,
                model='mock-v3',
                permission_mode='balanced',
            )

            exit_code = runtime.run()

            output = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertIn('Welcome to Anvil', output)
            self.assertIn(TOP_LEFT, output)
            self.assertEqual(output.count(TOP_LEFT), 1)
            self.assertIn(f'{PROMPT_MARKER} ', output)
            self.assertIn('? for shortcuts', output)
            self.assertIn('/status', output)
            self.assertIn('/model', output)
            self.assertIn('mock-v3', output)
            self.assertIn('requested: other-model', output)
            self.assertIn('Restart Anvil with --model', output)
            self.assertIn('balanced', output)
            self.assertIn(f'{RESPONSE_MARKER} done: hello', output)
            self.assertIn('Commands:', output)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_use_muted_terminal_theme_for_tty_output(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'chat-runtime-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            session_store = SessionStore.create(
                root_dir=tmp_dir / 'sessions',
                workspace_root=tmp_dir,
                goal='',
                memory_run_dir=tmp_dir / 'runs',
                session_id='sess-theme',
            )
            stdout = TtyStringIO()
            runtime = InteractiveRuntime(
                session_store=session_store,
                tool_specs=builtin_tool_specs(),
                run_turn=lambda text: f'done: {text}',
                stdin=io.StringIO('hello\n/exit\n'),
                stdout=stdout,
                model='mock-v3',
                permission_mode='balanced',
            )

            exit_code = runtime.run()

            output = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertIn(BORDER, output)
            self.assertIn(PROMPT, output)
            self.assertIn(ASSISTANT, output)
            self.assertNotIn('\033[36m', output)
            self.assertNotIn('\033[32m', output)
            self.assertNotIn('\033[35m', output)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_read_utf8_bytes_from_interactive_stdin(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'chat-runtime-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            request = '新增一个abc文件夹，并在abc新增一个空白的json文件'
            session_store = SessionStore.create(
                root_dir=tmp_dir / 'sessions',
                workspace_root=tmp_dir,
                goal='',
                memory_run_dir=tmp_dir / 'runs',
                session_id='sess-utf8',
            )
            calls: list[str] = []
            runtime = InteractiveRuntime(
                session_store=session_store,
                tool_specs=builtin_tool_specs(),
                run_turn=lambda text: calls.append(text) or f'done: {text}',
                stdin=Utf8BytesInput(f'{request}\n/exit\n'),
                stdout=io.StringIO(),
                model='mock-v3',
                permission_mode='balanced',
            )

            exit_code = runtime.run()

            self.assertEqual(exit_code, 0)
            self.assertEqual(calls, [request])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_save_conversation_transcript_to_txt(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'chat-runtime-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            session_store = SessionStore.create(
                root_dir=tmp_dir / 'sessions',
                workspace_root=tmp_dir,
                goal='',
                memory_run_dir=tmp_dir / 'runs',
                session_id='sess-save',
            )
            stdin = io.StringIO('你好\n保存目前的对话生成一个 txt 文件内容\n/exit\n')
            stdout = io.StringIO()
            calls: list[str] = []

            def run_turn(text: str) -> str:
                calls.append(text)
                return f'done: {text}'

            runtime = InteractiveRuntime(
                session_store=session_store,
                tool_specs=builtin_tool_specs(),
                run_turn=run_turn,
                stdin=stdin,
                stdout=stdout,
                model='mock-v3',
                permission_mode='balanced',
            )

            exit_code = runtime.run()

            transcript_path = tmp_dir / 'anvil-conversation-sess-save.txt'
            transcript = transcript_path.read_text(encoding='utf-8')
            output = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertEqual(calls, ['你好'])
            self.assertTrue(transcript_path.exists())
            self.assertIn('user: 你好', transcript)
            self.assertIn('assistant: done: 你好', transcript)
            self.assertIn('user: 保存目前的对话生成一个 txt 文件内容', transcript)
            self.assertIn('Saved conversation transcript to', output)
            self.assertIn(transcript_path.name, output)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_send_folder_request_to_model(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'chat-runtime-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            request = f'\u5728{tmp_dir.resolve()}\u65b0\u589e\u4e00\u4e2aabc\u6587\u4ef6\u5939'
            session_store = SessionStore.create(
                root_dir=tmp_dir / 'sessions',
                workspace_root=tmp_dir,
                goal='',
                memory_run_dir=tmp_dir / 'runs',
                session_id='sess-mkdir',
            )
            stdin = io.StringIO(f'{request}\n/exit\n')
            stdout = io.StringIO()
            calls: list[str] = []

            runtime = InteractiveRuntime(
                session_store=session_store,
                tool_specs=builtin_tool_specs(),
                run_turn=lambda text: calls.append(text) or f'done: {text}',
                stdin=stdin,
                stdout=stdout,
                model='mock-v3',
                permission_mode='balanced',
            )

            exit_code = runtime.run()

            target = tmp_dir / 'abc'
            output = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertEqual(calls, [request])
            self.assertFalse(target.exists())
            self.assertIn('done:', output)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_send_folder_and_file_request_to_model(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'chat-runtime-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            request = (
                f'\u5728{tmp_dir.resolve()}\u65b0\u589e\u4e00\u4e2aabc'
                '\uff0c\u5e76\u5728abc\u65b0\u589e\u4e00\u4e2a\u7a7a\u767d\u7684txt\u6587\u4ef6'
            )
            session_store = SessionStore.create(
                root_dir=tmp_dir / 'sessions',
                workspace_root=tmp_dir,
                goal='',
                memory_run_dir=tmp_dir / 'runs',
                session_id='sess-touch',
            )
            stdin = io.StringIO(f'{request}\n/exit\n')
            stdout = io.StringIO()
            calls: list[str] = []

            runtime = InteractiveRuntime(
                session_store=session_store,
                tool_specs=builtin_tool_specs(),
                run_turn=lambda text: calls.append(text) or f'done: {text}',
                stdin=stdin,
                stdout=stdout,
                model='mock-v3',
                permission_mode='balanced',
            )

            exit_code = runtime.run()

            target = tmp_dir / 'abc' / 'blank.txt'
            output = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertEqual(calls, [request])
            self.assertFalse(target.exists())
            self.assertIn('done:', output)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_parse_blank_file_fallback_extensions(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'chat-runtime-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            session_store = SessionStore.create(
                root_dir=tmp_dir / 'sessions',
                workspace_root=tmp_dir,
                goal='',
                memory_run_dir=tmp_dir / 'runs',
                session_id='sess-fallback-files',
            )
            runtime = InteractiveRuntime(
                session_store=session_store,
                tool_specs=builtin_tool_specs(),
                run_turn=lambda text: f'done: {text}',
                stdin=io.StringIO(''),
                stdout=io.StringIO(),
                model='mock-v3',
                permission_mode='balanced',
            )

            json_output = runtime._handle_filesystem_request(
                f'\u5728{tmp_dir.resolve()}\u65b0\u589e\u4e00\u4e2aabc'
                '\uff0c\u5e76\u5728abc\u65b0\u589e\u4e00\u4e2a\u7a7a\u767d\u7684json\u6587\u4ef6'
            )
            md_output = runtime._handle_filesystem_request(
                f'\u5728{tmp_dir.resolve()}\u65b0\u589e\u4e00\u4e2adefs'
                '\uff0c\u5e76\u5728defs\u65b0\u5efa\u4e00\u4e2a.md\u6587\u4ef6'
            )

            self.assertIsNotNone(json_output)
            self.assertIsNotNone(md_output)
            self.assertTrue((tmp_dir / 'abc' / 'blank.json').is_file())
            self.assertTrue((tmp_dir / 'defs' / 'blank.md').is_file())
            self.assertFalse((tmp_dir / 'abc' / 'json.txt').exists())
            self.assertFalse((tmp_dir / 'defs' / '.md.txt').exists())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_render_dynamic_working_status_for_tty(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'chat-runtime-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            session_store = SessionStore.create(
                root_dir=tmp_dir / 'sessions',
                workspace_root=tmp_dir,
                goal='',
                memory_run_dir=tmp_dir / 'runs',
                session_id='sess-dynamic',
            )
            stdout = TtyStringIO()
            runtime = InteractiveRuntime(
                session_store=session_store,
                tool_specs=builtin_tool_specs(),
                run_turn=lambda text: f'done: {text}',
                stdin=io.StringIO(''),
                stdout=stdout,
                model='mock-v3',
                permission_mode='balanced',
            )

            result = runtime._run_with_working_status(lambda: time.sleep(0.2) or 'done')

            output = stdout.getvalue()
            self.assertEqual(result, 'done')
            self.assertIn('\r', output)
            self.assertIn('\033[?25l', output)
            self.assertIn('\033[?25h', output)
            self.assertIn('\033[2K', output)
            self.assertIn('Working...', output)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
