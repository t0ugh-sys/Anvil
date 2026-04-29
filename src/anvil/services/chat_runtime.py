from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, TextIO

from ..commands.slash import execute_slash_command, parse_slash_command
from ..messages import AssistantMessage, UserMessage
from ..session import SessionStore
from ..tool_spec import ToolSpec


TurnRunner = Callable[[str], str]


@dataclass
class InteractiveRuntime:
    session_store: SessionStore
    tool_specs: Iterable[ToolSpec]
    run_turn: TurnRunner
    runtime_config_manager: object | None
    stdin: TextIO
    stdout: TextIO

    def run(self) -> int:
        self._write_line(
            f'Anvil interactive session {self.session_store.state.session_id} '
            f'({self.session_store.state.workspace_root})'
        )
        self._write_line('Type /help for commands. Use /provider and /model to switch models during the session.')
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
        output = self.run_turn(text).strip() or 'No response.'
        assistant_message = AssistantMessage(content=output)
        self.session_store.append_event(
            'chat_assistant',
            {'role': assistant_message.role, 'content': assistant_message.content},
        )
        self._write_line(assistant_message.content)

    def _write(self, value: str) -> None:
        self.stdout.write(value)
        self.stdout.flush()

    def _write_line(self, value: str) -> None:
        self.stdout.write(value + '\n')
        self.stdout.flush()
