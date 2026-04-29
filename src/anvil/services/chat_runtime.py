from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, TextIO

from ..commands.slash import execute_slash_command, parse_slash_command, render_session_header
from ..messages import AssistantMessage, UserMessage
from ..session import SessionStore
from ..tool_spec import ToolSpec


@dataclass(frozen=True)
class TurnExecution:
    output: str
    stop_reason: str = ''
    steps: int = 0
    permission_delta: dict[str, int] | None = None
    blocked_tool_name: str = ''
    blocked_tool_reason: str = ''


TurnRunner = Callable[[str], TurnExecution | str]


@dataclass
class InteractiveRuntime:
    session_store: SessionStore
    tool_specs: Iterable[ToolSpec]
    run_turn: TurnRunner
    runtime_config_manager: object | None
    stdin: TextIO
    stdout: TextIO

    def run(self) -> int:
        self._write_line(render_session_header(self.session_store.state, runtime_label='interactive'))
        self._write_line(
            'Type /help for commands. Use /status to inspect the session. '
            'Use /provider and /model to switch models during the session.'
        )
        while True:
            self._write('anvil> ')
            line = self.stdin.readline()
            if line == '':
                self._write_line('')
                return 0
            text = line.strip()
            if not text:
                continue
            command = parse_slash_command(text)
            if command is not None:
                result = execute_slash_command(
                    command,
                    session_store=self.session_store,
                    tool_specs=self.tool_specs,
                    runtime_config_manager=self.runtime_config_manager,
                )
                self.session_store.append_event('chat_command', {'command': command.name, 'argument': command.argument})
                self._write_line(result.output)
                if not result.should_continue:
                    return 0
                continue
            self._handle_message(text)

    def _handle_message(self, text: str) -> None:
        user_message = UserMessage(content=text)
        self.session_store.append_event('chat_user', {'role': user_message.role, 'content': user_message.content})
        turn_execution = self._coerce_turn_execution(self.run_turn(text))
        output = turn_execution.output.strip() or 'No response.'
        assistant_message = AssistantMessage(content=output)
        self.session_store.append_event(
            'chat_assistant',
            {'role': assistant_message.role, 'content': assistant_message.content},
        )
        self._write_line(assistant_message.content)
        footer = self._render_turn_footer(turn_execution)
        if footer:
            self._write_line(footer)

    def _coerce_turn_execution(self, value: TurnExecution | str) -> TurnExecution:
        if isinstance(value, TurnExecution):
            return value
        return TurnExecution(output=str(value))

    def _render_turn_footer(self, turn_execution: TurnExecution) -> str:
        parts: list[str] = []
        if turn_execution.stop_reason:
            parts.append(f'stop_reason={turn_execution.stop_reason}')
        if turn_execution.steps > 0:
            parts.append(f'steps={turn_execution.steps}')
        permission_delta = turn_execution.permission_delta or {}
        permission_parts = [
            f'{name}+{count}'
            for name, count in permission_delta.items()
            if isinstance(count, int) and count > 0
        ]
        if permission_parts:
            parts.append('permissions=' + ','.join(permission_parts))
        if turn_execution.blocked_tool_name:
            blocked = turn_execution.blocked_tool_name
            if turn_execution.blocked_tool_reason:
                blocked += f' ({turn_execution.blocked_tool_reason})'
            parts.append(f'blocked={blocked}')
        state = self.session_store.state
        parts.append(
            f'session=turns:{state.turn_count} messages:{state.message_count} '
            f'commands:{state.command_count} steps:{state.step_count}'
        )
        return '[turn] ' + ' '.join(parts) if parts else ''

    def _write(self, value: str) -> None:
        self.stdout.write(value)
        self.stdout.flush()

    def _write_line(self, value: str) -> None:
        self.stdout.write(value + '\n')
        self.stdout.flush()
