from __future__ import annotations

import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, TextIO, TypeVar

from ..commands.slash import execute_slash_command, parse_slash_command
from ..messages import AssistantMessage, UserMessage
from ..session import SessionStore
from ..tool_spec import ToolSpec
from ..ui.chrome import (
    ACCENT,
    ASSISTANT,
    BORDER,
    DIM,
    DOT_SEPARATOR,
    PROMPT_MARKER,
    PROMPT,
    RESPONSE_MARKER,
    WORKING_MARKER,
    WORKING,
    bounded_width,
    box_lines,
    colorize,
    response_lines,
    truncate,
)


TurnRunner = Callable[[str], str]
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

    def run(self) -> int:
        self.tool_specs = tuple(self.tool_specs)
        width = self._ui_width()
        self._write_welcome(width)
        while True:
            self._write_prompt()
            line = self._read_input_line()
            if line == '':
                self._write_line('')
                return 0

            text = line.strip()
            if not text:
                continue

            command = parse_slash_command('/help' if text == '?' else text)
            if command is not None:
                if command.name == 'model':
                    self.session_store.append_event(
                        'chat_command',
                        {'command': command.name, 'argument': command.argument},
                    )
                    self._write_response(self._model_command_output(command.argument), width=width)
                    continue
                result = execute_slash_command(
                    command,
                    session_store=self.session_store,
                    tool_specs=self.tool_specs,
                )
                self.session_store.append_event('chat_command', {'command': command.name, 'argument': command.argument})
                self._write_response(result.output, width=width)
                if not result.should_continue:
                    return 0
                continue
            self._handle_message(text)

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

    def _run_with_working_status(self, operation: Callable[[], T], *, minimum_s: float = 0.0) -> T:
        started_at = time.monotonic()
        if not self._color_enabled():
            self._write_line(self._style(f'  {WORKING_MARKER} Working...', WORKING))
            result = operation()
            self._wait_for_minimum_status_duration(started_at, minimum_s)
            return result
        status = _WorkingStatus(self.stdout, self._ui_width(), style=WORKING)
        status.start()
        try:
            return operation()
        finally:
            self._wait_for_minimum_status_duration(started_at, minimum_s)
            status.stop()

    def _handle_filesystem_request(self, text: str) -> str | None:
        return self._create_blank_file_from_request(text) or self._create_directory_from_request(text)

    def _wait_for_minimum_status_duration(self, started_at: float, minimum_s: float) -> None:
        remaining = minimum_s - (time.monotonic() - started_at)
        if remaining > 0:
            time.sleep(remaining)

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
            r'^\s*(?:\u5728)?(?P<base>(?:[A-Za-z]:[\\/]|[\\/])[^,\uff0c]*?)\s*'
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
            r'^\s*(?:\u5728)?(?P<base>(?:[A-Za-z]:[\\/]|[\\/])[^,\uff0c]*?)\s*'
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
        lines = [
            f'{WORKING_MARKER} Welcome to Anvil',
            '',
            f'cwd: {state.workspace_root}',
            f'session: {state.session_id}',
            f'model: {model}',
            f'permissions: {permission_mode}',
            f'tools: {tool_count}',
        ]
        for index, line in enumerate(box_lines(lines, width=width, title='Anvil')):
            style = BORDER if index in {0, len(lines) + 1} else DIM
            self._write_line(self._style(line, style))
        hint = f'  ? for shortcuts {DOT_SEPARATOR} /help {DOT_SEPARATOR} /status {DOT_SEPARATOR} /model {DOT_SEPARATOR} /panel {DOT_SEPARATOR} /exit'
        self._write_line(self._style(truncate(hint, width), DIM))
        self._write_line(self._style(self._status_line(width), DIM))
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

    def _read_input_line(self) -> str:
        buffer = getattr(self.stdin, 'buffer', None)
        if buffer is not None:
            try:
                raw = buffer.readline()
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

    def _write_response(self, value: str, *, width: int) -> None:
        for line in response_lines(value, width=width):
            if line.lstrip().startswith(RESPONSE_MARKER):
                marker, _, rest = line.partition(RESPONSE_MARKER)
                rendered = marker + self._style(RESPONSE_MARKER, ACCENT) + self._style(rest, ASSISTANT)
            else:
                rendered = self._style(line, ASSISTANT)
            self._write_line(rendered)
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
        return bool(getattr(self.stdout, 'isatty', lambda: False)())

    def _write(self, value: str) -> None:
        self.stdout.write(value)
        self.stdout.flush()

    def _write_line(self, value: str) -> None:
        self.stdout.write(value + '\n')
        self.stdout.flush()


class _WorkingStatus:
    _FRAMES = ('\u2736', '\u2737', '\u2738', '\u2739', '\u273a', '\u273b')

    def __init__(self, stdout: TextIO, width: int, *, style: str = '') -> None:
        self._stdout = stdout
        self._width = width
        self._style = style
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stdout.write('\033[?25l')
        self._stdout.flush()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        self._clear_line()

    def _spin(self) -> None:
        index = 0
        while not self._stop_event.is_set():
            frame = self._FRAMES[index % len(self._FRAMES)]
            line = truncate(f'  {frame} Working...', self._width)
            if self._style:
                line = colorize(line, self._style, enabled=True)
            self._stdout.write('\r\033[2K' + line)
            self._stdout.flush()
            index += 1
            self._stop_event.wait(0.12)

    def _clear_line(self) -> None:
        self._stdout.write('\r\033[2K\033[?25h')
        self._stdout.flush()
