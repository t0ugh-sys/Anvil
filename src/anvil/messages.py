from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


MessageRole = Literal['system', 'user', 'assistant']


@dataclass(frozen=True)
class ChatMessage:
    role: MessageRole
    content: str

    def render_line(self) -> str:
        return f'{self.role}: {self.content}'


class SystemMessage(ChatMessage):
    def __init__(self, content: str) -> None:
        super().__init__(role='system', content=content)


class UserMessage(ChatMessage):
    def __init__(self, content: str) -> None:
        super().__init__(role='user', content=content)


class AssistantMessage(ChatMessage):
    def __init__(self, content: str) -> None:
        super().__init__(role='assistant', content=content)
