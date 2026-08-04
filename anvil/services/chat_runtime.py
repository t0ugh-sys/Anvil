from __future__ import annotations

import asyncio
import sys
import re
import shutil
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable, Iterable, TextIO, TypeVar

if TYPE_CHECKING:
    from prompt_toolkit import PromptSession

from ..commands.slash import execute_slash_command, parse_slash_command
from ..infra.permissions import set_ask_permission_fn
from ..messages import AssistantMessage, UserMessage
from ..runtime.session import SessionStore
from ..tool_spec import ToolSpec
from ..ui.chrome import (
    ACCENT,
    ASSISTANT,
    BORDER,
    BOTTOM_LEFT,
    BOTTOM_RIGHT,
    CIRCLE_EMPTY,
    DIM,
    DIAMOND,
    DOT_SEPARATOR,
    DOUBLE_HORIZONTAL,
    DOUBLE_VERTICAL,
    GEAR,
    HORIZONTAL,
    LIGHTNING,
    MUTED,
    PROMPT_MARKER,
    PROMPT,
    RESPONSE_MARKER,
    SPARKLE,
    TOP_LEFT,
    TOP_RIGHT,
    VERTICAL,
    WORKING_MARKER,
    WORKING,
    bounded_width,
    box_lines,
    colorize,
    response_lines,
    separator_line,
    truncate,
    wrap_line,
)
from ..ui.tool_renderer import set_line_writer

__all__ = ['InteractiveRuntime']

# Shared regex fragment for matching an absolute path prefix (e.g. C:\... or /...)
_BASE_PATH_PATTERN = r'(?:[A-Za-z]:[\\/]|[\\/])[^,\uff0c]*?'


TurnRunner = Callable[[str], str] | Callable[[str], Awaitable[str]]
T = TypeVar('T')


@dataclass
class InteractiveRuntime:
    session_store: SessionStore
    tool_specs: Iterable[ToolSpec]
    run_turn: TurnRunner
    stdin: TextIO
    stdout: TextIO
    model: str = ''
    permission_mode: str = ''
    usage_tracker: object = None
    rate_limit_tracker: object = None

    def run(self) -> int:
        set_line_writer(self._write_tool_line)
        try:
            return asyncio.run(self._run_async())
        finally:
            set_line_writer(None)

    async def _run_async(self) -> int:
        is_tty = bool(getattr(self.stdin, 'isatty', lambda: False)())
        if is_tty and self.stdout is sys.stdout:
            # Keep background agent/tool output above an active prompt. The
            # proxy also serializes writes coming from the worker thread.
            from prompt_toolkit.patch_stdout import patch_stdout

            original_stdout = self.stdout
            # Keep terminal control sequences intact. ``raw=False`` replaces
            # ESC with ``?`` on Vt100 output, which leaks sequences such as
            # ``?[2K`` into Windows terminals and prevents colors from being
            # rendered.
            with patch_stdout(raw=True):
                self.stdout = sys.stdout
                self._prompt_toolkit_output = True
                try:
                    return await self._run_loop_async()
                finally:
                    self._prompt_toolkit_output = False
                    self.stdout = original_stdout
        return await self._run_loop_async()

    async def _run_loop_async(self) -> int:
        self.tool_specs = tuple(self.tool_specs)
        is_tty = bool(getattr(self.stdin, 'isatty', lambda: False)())
        if is_tty:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.history import InMemoryHistory
            self._prompt_session: PromptSession | None = PromptSession(history=InMemoryHistory())
        else:
            self._prompt_session = None

        width = self._ui_width()
        self._write_welcome(width)

        while True:
            if self._prompt_session is None:
                self._write_prompt()
            try:
                line = await self._read_input_line_async()
            except (EOFError, KeyboardInterrupt):
                self._write_line('')
                return 0
            if line == '':
                self._write_line('')
                return 0

            text = line.strip()
            if not text:
                continue

            should_continue, queued = await self._dispatch_line_async(text, width=width)
            if not should_continue:
                return 0
            while queued:
                queued_text = queued.pop(0)
                should_continue, new_queue = await self._dispatch_line_async(queued_text, width=width)
                if not should_continue:
                    return 0
                queued.extend(new_queue)

    async def _dispatch_line_async(self, text: str, *, width: int) -> tuple[bool, list[str]]:
        command = parse_slash_command('/help' if text == '?' else text)
        if command is not None:
            if command.name == 'model':
                self.session_store.append_event(
                    'chat_command',
                    {'command': command.name, 'argument': command.argument},
                )
                self._write_response(self._model_command_output(command.argument), width=width)
                return True, []
            result = execute_slash_command(
                command,
                session_store=self.session_store,
                tool_specs=self.tool_specs,
                usage_tracker=self.usage_tracker,
                rate_limit_tracker=self.rate_limit_tracker,
            )
            self.session_store.append_event('chat_command', {'command': command.name, 'argument': command.argument})
            self._write_response(result.output, width=width)
            return result.should_continue, []

        return True, await self._handle_message_with_input_async(text)

    async def _handle_message_with_input_async(self, text: str) -> list[str]:
        """Run one agent turn while accepting later input into a FIFO queue."""
        if self._prompt_session is None:
            await self._handle_message_async(text)
            return []

        agent_task = asyncio.create_task(self._handle_message_async(text))
        # The working indicator is rendered by the prompt layout, so the
        # prompt can open immediately while the agent continues running.
        await asyncio.sleep(0)
        input_task: asyncio.Task[str] | None = asyncio.create_task(self._read_input_line_async())
        queued: list[str] = []

        while True:
            wait_for = {agent_task}
            if input_task is not None:
                wait_for.add(input_task)
            done, _ = await asyncio.wait(wait_for, return_when=asyncio.FIRST_COMPLETED)

            if input_task is not None and input_task in done:
                try:
                    line = input_task.result()
                except (EOFError, KeyboardInterrupt):
                    self._input_closed = True
                    input_task = None
                else:
                    if line.strip():
                        queued.append(line.strip())
                    if not agent_task.done() and not getattr(self, '_input_closed', False):
                        input_task = asyncio.create_task(self._read_input_line_async())
                    else:
                        input_task = None

            if agent_task in done:
                break

        if input_task is not None:
            input_task.cancel()
            with suppress(asyncio.CancelledError):
                await input_task
        await agent_task
        return queued

    def _handle_message(self, text: str) -> None:
        user_message = UserMessage(content=text)
        self.session_store.append_event('chat_user', {'role': user_message.role, 'content': user_message.content})
        if self._is_save_conversation_request(text):
            output = self._run_with_working_status(self._save_conversation_transcript)
        else:
            output = (
                self._run_with_working_status(lambda: self.run_turn(text)).strip()
                or 'Stopped without final output.'
            )
        assistant_message = AssistantMessage(content=output)
        self.session_store.append_event(
            'chat_assistant',
            {'role': assistant_message.role, 'content': assistant_message.content},
        )
        self._write_response(assistant_message.content, width=self._ui_width())

    async def _handle_message_async(self, text: str) -> None:
        user_message = UserMessage(content=text)
        self.session_store.append_event('chat_user', {'role': user_message.role, 'content': user_message.content})
        if self._is_save_conversation_request(text):
            loop = asyncio.get_event_loop()
            output = await self._run_with_working_status_async(
                lambda: loop.run_in_executor(None, self._save_conversation_transcript)
            )
        else:
            async def _turn() -> str:
                raw = self.run_turn(text)
                if asyncio.iscoroutine(raw):
                    return await raw
                return raw  # type: ignore[return-value]

            output = (await self._run_with_working_status_async(_turn)).strip() or 'Stopped without final output.'
        assistant_message = AssistantMessage(content=output)
        self.session_store.append_event(
            'chat_assistant',
            {'role': assistant_message.role, 'content': assistant_message.content},
        )
        self._write_response(assistant_message.content, width=self._ui_width())

    async def _run_with_working_status_async(self, coro_fn: Callable[[], Awaitable[T]], *, minimum_s: float = 0.0) -> T:
        started_at = time.monotonic()
        prompt_toolkit_active = getattr(self, '_prompt_session', None) is not None
        ansi_enabled = self._color_enabled()
        if ansi_enabled or prompt_toolkit_active:
            status: _WorkingStatus | None = _WorkingStatus(
                self.stdout,
                self._ui_width(),
                style=WORKING,
                use_ansi=ansi_enabled,
                render_in_prompt=prompt_toolkit_active and getattr(self, '_prompt_toolkit_output', False),
            )
            status.start()
        else:
            status = None
            working_line = f'  {WORKING_MARKER} Working...'
            if not prompt_toolkit_active:
                working_line = self._style(working_line, WORKING)
            self._write_line(working_line)
        self._active_status = status
        set_ask_permission_fn(self._build_ask_fn(status))
        try:
            result = await coro_fn()
        finally:
            set_ask_permission_fn(None)
            self._wait_for_minimum_status_duration(started_at, minimum_s)
            if status is not None:
                status.stop()
            self._active_status = None
        return result

    def _run_with_working_status(self, operation: Callable[[], T], *, minimum_s: float = 0.0) -> T:
        started_at = time.monotonic()
        prompt_toolkit_active = getattr(self, '_prompt_session', None) is not None
        ansi_enabled = self._color_enabled()
        if ansi_enabled or prompt_toolkit_active:
            status = _WorkingStatus(
                self.stdout,
                self._ui_width(),
                style=WORKING,
                use_ansi=ansi_enabled,
                render_in_prompt=prompt_toolkit_active and getattr(self, '_prompt_toolkit_output', False),
            )
            status.start()
            self._active_status = status
            set_ask_permission_fn(self._build_ask_fn(status))
            try:
                return operation()
            finally:
                set_ask_permission_fn(None)
                self._wait_for_minimum_status_duration(started_at, minimum_s)
                status.stop()
                self._active_status = None

        if prompt_toolkit_active or not self._color_enabled():
            working_line = f'  {WORKING_MARKER} Working...'
            if not prompt_toolkit_active:
                working_line = self._style(working_line, WORKING)
            self._write_line(working_line)
            self._active_status = None
            set_ask_permission_fn(self._build_ask_fn(None))
            try:
                result = operation()
            finally:
                set_ask_permission_fn(None)
            self._wait_for_minimum_status_duration(started_at, minimum_s)
            return result

    def _handle_filesystem_request(self, text: str) -> str | None:
        return self._create_blank_file_from_request(text) or self._create_directory_from_request(text)

    def _wait_for_minimum_status_duration(self, started_at: float, minimum_s: float) -> None:
        remaining = minimum_s - (time.monotonic() - started_at)
        if remaining > 0:
            time.sleep(remaining)

    def _start_input_reader(self) -> None:
        pass  # replaced by PromptSession

    def _raw_read_stdin(self) -> str:
        buf = getattr(self.stdin, 'buffer', None)
        if buf is not None:
            try:
                raw = buf.readline()
            except Exception:
                raw = None
            if isinstance(raw, bytes):
                if raw == b'':
                    return ''
                try:
                    return raw.decode('utf-8')
                except UnicodeDecodeError:
                    return raw.decode('mbcs', errors='replace')
        return self.stdin.readline()

    def _build_ask_fn(self, status: _WorkingStatus | None) -> Callable[[str, dict], bool]:
        def ask_fn(tool_name: str, arguments: dict) -> bool:
            if status is not None:
                status.pause()
            try:
                args_preview = ', '.join(
                    f'{k}={repr(v)[:40]}' for k, v in list(arguments.items())[:3]
                )
                self._write(f'\n  Allow {tool_name}({args_preview})? [Enter=yes / n=no]: ')
                line = self._raw_read_stdin()
                return line.strip().lower() not in {'n', 'no'}
            finally:
                if status is not None:
                    status.resume()
        return ask_fn

    def _is_save_conversation_request(self, text: str) -> bool:
        normalized = text.lower()
        wants_save = any(
            token in normalized
            for token in ('\u4fdd\u5b58', '\u5199\u5230', '\u5199\u5165', '\u5bfc\u51fa')
        )
        mentions_chat = any(
            token in normalized
            for token in ('\u5bf9\u8bdd', '\u804a\u5929', 'conversation', 'chat')
        )
        mentions_text_file = any(
            token in normalized
            for token in ('txt', '.txt', '\u6587\u672c', '\u6587\u4ef6')
        )
        return wants_save and mentions_chat and mentions_text_file

    def _create_directory_from_request(self, text: str) -> str | None:
        match = re.match(
            r'^\s*(?:\u5728)?(?P<base>' + _BASE_PATH_PATTERN + r')\s*'
            r'(?:\u65b0\u589e|\u521b\u5efa|\u65b0\u5efa)\s*(?:\u4e00\u4e2a)?'
            r'(?P<name>[^\\/:*?"<>|\s,\uff0c]+)\s*(?:\u6587\u4ef6\u5939|\u76ee\u5f55)\s*$',
            text,
            flags=re.IGNORECASE,
        )
        if match is None:
            return None
        base = Path(match.group('base')).resolve()
        name = match.group('name').strip()
        target = (base / name).resolve()
        if target.exists() and not target.is_dir():
            return f'Cannot create folder because a file already exists: {target}'
        if target.exists():
            return f'Folder already exists: {target}'
        target.mkdir(parents=True, exist_ok=True)
        return f'Created folder: {target}'

    def _create_blank_file_from_request(self, text: str) -> str | None:
        match = re.match(
            r'^\s*(?:\u5728)?(?P<base>' + _BASE_PATH_PATTERN + r')\s*'
            r'(?:\u65b0\u589e|\u521b\u5efa|\u65b0\u5efa)\s*(?:\u4e00\u4e2a)?'
            r'(?P<folder>[^\\/:*?"<>|\s,\uff0c]+)\s*(?:\u6587\u4ef6\u5939|\u76ee\u5f55)?'
            r'\s*(?:,|\uff0c|\u5e76|\u5e76\u4e14|\u7136\u540e|\u518d)*\s*'
            r'(?:\u5728)?(?P=folder)(?:\u91cc|\u4e2d|\u4e0b|\u4e0b\u9762)?\s*'
            r'(?:\u65b0\u589e|\u521b\u5efa|\u65b0\u5efa)\s*(?:\u4e00\u4e2a)?'
            r'(?P<file_spec>.*?)\s*\u6587\u4ef6\s*$',
            text,
            flags=re.IGNORECASE,
        )
        if match is None:
            return None
        base = Path(match.group('base')).resolve()
        folder = match.group('folder').strip()
        folder_path = (base / folder).resolve()
        if folder_path.exists() and not folder_path.is_dir():
            return f'Cannot create folder because a file already exists: {folder_path}'
        folder_path.mkdir(parents=True, exist_ok=True)

        filename = self._blank_filename_from_spec(match.group('file_spec') or '')
        target = (folder_path / filename).resolve()
        if target.exists() and target.is_dir():
            return f'Cannot create file because a folder already exists: {target}'
        if target.exists():
            return f'Folder already exists: {folder_path}\nFile already exists: {target}'
        target.write_text('', encoding='utf-8')
        return f'Created folder: {folder_path}\nCreated file: {target}'

    def _blank_filename_from_spec(self, value: str) -> str:
        spec = value.strip().lower()
        for token in ('\u4e00\u4e2a', '\u7a7a\u767d\u7684', '\u7a7a\u767d', '\u7a7a\u7684', '\u7a7a'):
            spec = spec.replace(token, '')
        spec = spec.strip()
        if spec in {'', 'txt', '.txt'}:
            return 'blank.txt'
        if spec in {'json', '.json'}:
            return 'blank.json'
        if spec in {'md', '.md', 'markdown', '.markdown'}:
            return 'blank.md'
        if spec.startswith('.') and len(spec) > 1:
            return f'blank{spec}'
        return spec

    def _save_conversation_transcript(self) -> str:
        workspace_root = Path(self.session_store.state.workspace_root).resolve()
        workspace_root.mkdir(parents=True, exist_ok=True)
        session_id = self._safe_filename(self.session_store.state.session_id)
        transcript_path = workspace_root / f'anvil-conversation-{session_id}.txt'
        lines = [
            'Anvil conversation transcript',
            f'session: {self.session_store.state.session_id}',
            f'workspace: {self.session_store.state.workspace_root}',
            '',
        ]
        lines.extend(self.session_store.state.history_tail)
        transcript_path.write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')
        return f'Saved conversation transcript to {transcript_path}'

    def _safe_filename(self, value: str) -> str:
        cleaned = ''.join(char if char.isalnum() or char in {'-', '_', '.'} else '_' for char in value)
        return cleaned.strip('._') or 'session'

    def _ui_width(self) -> int:
        return bounded_width(shutil.get_terminal_size((88, 24)).columns)

    def _write_welcome(self, width: int) -> None:
        state = self.session_store.state
        tool_count = len(tuple(self.tool_specs))
        model = self.model.strip() or 'configured model'
        permission_mode = self.permission_mode.strip() or 'balanced'
        color = self._color_enabled()

        inner = max(2, width - 2)
        content_width = max(1, width - 4)  # │ + space + content + space + │

        def _row(text: str, style: str) -> str:
            padded = truncate(text, content_width).ljust(content_width)
            if color:
                return (
                    self._style(VERTICAL, BORDER)
                    + ' ' + colorize(padded, style, enabled=True) + ' '
                    + self._style(VERTICAL, BORDER)
                )
            return f'{VERTICAL} {padded} {VERTICAL}'

        self._write_line(self._style(TOP_LEFT + HORIZONTAL * inner + TOP_RIGHT, BORDER))
        self._write_line(_row(f'  {WORKING_MARKER} Anvil', ACCENT))
        self._write_line(_row('', DIM))
        self._write_line(_row(f'  {state.workspace_root}', ASSISTANT))
        info = f'  {model}  {DOT_SEPARATOR}  {permission_mode}  {DOT_SEPARATOR}  {tool_count} tools'
        self._write_line(_row(info, DIM))
        self._write_line(self._style(BOTTOM_LEFT + HORIZONTAL * inner + BOTTOM_RIGHT, BORDER))

        hint = f'  /help for help  {DOT_SEPARATOR}  /status  {DOT_SEPARATOR}  /exit'
        self._write_line(self._style(truncate(hint, width), DIM))
        self._write_line('')

    def _model_command_output(self, argument: str) -> str:
        model = self.model.strip() or 'model'
        requested = argument.strip()
        if not requested:
            return f'model: {model}'
        return (
            f'model: {model}\n'
            f'requested: {requested}\n'
            'Switching models inside the current interactive session is not supported yet. '
            'Restart Anvil with --model to change it.'
        )

    def _write_prompt(self) -> None:
        marker = self._style(PROMPT_MARKER, PROMPT)
        self._write(f'{marker} ')

    def _input_box_prompt(self) -> str:
        """Boxed prompt string — top border + left bar, matching Claude Code's input area."""
        width = self._ui_width()
        h = '─' * max(0, width - 2)
        top = self._style(f'╭{h}╮', BORDER)
        bar = self._style('│', BORDER)
        marker = self._style(PROMPT_MARKER, PROMPT)
        return f'{top}\n{bar} {marker} '

    def _input_box_prompt_dynamic(self, status: '_WorkingStatus | None'):
        """Build a prompt message whose working marker refreshes in place."""
        from prompt_toolkit.formatted_text import ANSI

        width = self._ui_width()
        horizontal = HORIZONTAL * max(0, width - 2)

        def render():
            working = ''
            if status is not None and status.running:
                working = self._style(f'  {status.frame} Working...\n', WORKING)
            top = self._style(f'{TOP_LEFT}{horizontal}{TOP_RIGHT}', BORDER)
            bar = self._style(VERTICAL, BORDER)
            marker = self._style(PROMPT_MARKER, PROMPT)
            return ANSI(f'{working}{top}\n{bar} {marker} ')

        return render

    def _input_box_toolbar(self) -> str:
        """Bottom border for the boxed input area."""
        width = self._ui_width()
        h = '─' * max(0, width - 2)
        return self._style(f'╰{h}╯', BORDER)

    def _read_input_line(self) -> str:
        if self._prompt_session is not None:
            status = getattr(self, '_active_status', None)
            message = self._input_box_prompt_dynamic(status)
            kwargs = {'refresh_interval': 0.12} if status is not None else {}
            result = self._prompt_session.prompt(message, **kwargs)
            # Close the box in terminal history after user submits.
            self._write_line(self._input_box_toolbar())
            return result
        return self._raw_read_stdin()

    async def _read_input_line_async(self) -> str:
        if self._prompt_session is not None:
            status = getattr(self, '_active_status', None)
            message = self._input_box_prompt_dynamic(status)
            kwargs = {'refresh_interval': 0.12} if status is not None else {}
            result = await self._prompt_session.prompt_async(message, **kwargs)
            # Close the box in terminal history after user submits.
            self._write_line(self._input_box_toolbar())
            return result
        # Non-TTY: run blocking readline in thread pool so event loop stays free.
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._raw_read_stdin)

    def _write_response(self, value: str, *, width: int) -> None:
        content = value.strip() or 'No response.'
        content_width = max(1, width - 4)
        for raw_line in content.splitlines() or ['']:
            for wrapped in wrap_line(raw_line, content_width):
                self._write_line(self._style('  ' + wrapped, ASSISTANT))
        self._write_line('')
        self._write_line(self._style('  ' + separator_line(width - 4), BORDER))
        self._write_line(self._style(self._status_line(width), DIM))
        self._write_line('')

    def _status_line(self, width: int) -> str:
        state = self.session_store.state
        parts = [
            self.model.strip() or 'model',
            self.permission_mode.strip() or 'balanced',
            state.workspace_root,
        ]
        return truncate('  ' + f' {DOT_SEPARATOR} '.join(parts), width)

    def _style(self, value: str, style: str) -> str:
        return colorize(value, style, enabled=self._color_enabled())

    def _color_enabled(self) -> bool:
        # The patched prompt_toolkit stdout is raw-aware, so ANSI is safe for
        # the real terminal. Keep manually constructed test prompt sessions
        # plain because they do not have that output proxy.
        if getattr(self, '_prompt_toolkit_output', False):
            return bool(getattr(self.stdout, 'isatty', lambda: False)())
        if getattr(self, '_prompt_session', None) is not None:
            return False
        return bool(getattr(self.stdout, 'isatty', lambda: False)())

    def _write(self, value: str) -> None:
        self.stdout.write(value)
        self.stdout.flush()

    def _write_line(self, value: str) -> None:
        self.stdout.write(value + '\n')
        self.stdout.flush()

    def _write_tool_line(self, value: str) -> None:
        """Print tool output without leaving the working line underneath it."""
        status = getattr(self, '_active_status', None)
        if status is not None:
            status.pause()
        self._write_line(value)
        if status is not None:
            status.resume()


class _WorkingStatus:
    _FRAMES = ('\u2736', '\u2737', '\u2738', '\u2739', '\u273a', '\u273b')

    def __init__(
        self,
        stdout: TextIO,
        width: int,
        *,
        style: str = '',
        use_ansi: bool = True,
        render_in_prompt: bool = False,
    ) -> None:
        self._stdout = stdout
        self._width = width
        self._style = style
        self._use_ansi = use_ansi
        self._render_in_prompt = render_in_prompt
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._held_for_input = threading.Event()
        self._frame_index = 0
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return not self._stop_event.is_set()

    @property
    def frame(self) -> str:
        return self._FRAMES[self._frame_index % len(self._FRAMES)]

    def start(self) -> None:
        if self._use_ansi and not self._render_in_prompt:
            self._stdout.write('\033[?25l')
        self._stdout.flush()
        if not self._render_in_prompt:
            self._render_frame(0)
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if not self._held_for_input.is_set() and not self._render_in_prompt:
            self._clear_line()

    def hold_for_input(self) -> None:
        """Commit the current status line before opening an input prompt."""
        if self._render_in_prompt:
            return
        if self._held_for_input.is_set():
            return
        self._paused.set()
        self._held_for_input.set()
        if self._use_ansi:
            # The spinner owns the cursor while it is active. Return it to the
            # prompt before committing the status line to terminal history.
            self._stdout.write('\033[?25h')
        self._stdout.write('\n')
        self._stdout.flush()

    def pause(self) -> None:
        if self._render_in_prompt:
            self._paused.set()
            return
        if self._held_for_input.is_set():
            return
        self._paused.set()
        if self._use_ansi:
            self._stdout.write('\r\033[2K\033[?25h')
        else:
            self._stdout.write('\r' + (' ' * self._width) + '\r')
        self._stdout.flush()

    def resume(self) -> None:
        if self._render_in_prompt:
            self._paused.clear()
            return
        if self._held_for_input.is_set():
            return
        self._paused.clear()
        if self._use_ansi:
            self._stdout.write('\033[?25l')
        self._stdout.flush()

    def _render_frame(self, index: int) -> None:
        frame = self._FRAMES[index % len(self._FRAMES)]
        line = truncate(f'  {frame} Working...', self._width)
        if self._style and self._use_ansi:
            line = colorize(line, self._style, enabled=True)
        if self._use_ansi:
            self._stdout.write('\r\033[2K' + line)
        else:
            self._stdout.write('\r' + line.ljust(self._width))
        self._stdout.flush()

    def _spin(self) -> None:
        index = 0
        while not self._stop_event.is_set():
            if not self._paused.is_set():
                self._frame_index = index
                if not self._render_in_prompt:
                    self._render_frame(index)
                index += 1
            self._stop_event.wait(0.12)

    def _clear_line(self) -> None:
        if self._use_ansi:
            self._stdout.write('\r\033[2K\033[?25h')
        else:
            self._stdout.write('\r' + (' ' * self._width) + '\r')
        self._stdout.flush()
