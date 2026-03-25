from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class MailMessage:
    id: str
    sender: str
    recipient: str
    subject: str
    body: str
    task_id: str | None = None
    created_at: str = ''
    metadata: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'sender': self.sender,
            'recipient': self.recipient,
            'subject': self.subject,
            'body': self.body,
            'task_id': self.task_id,
            'created_at': self.created_at,
            'metadata': self.metadata or {},
        }


class JsonlMailbox:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._messages_file = self.root_dir / 'messages.jsonl'

    def send(self, message: MailMessage) -> None:
        payload = message.to_dict()
        if not payload['created_at']:
            payload['created_at'] = _utc_now()
        with self._messages_file.open('a', encoding='utf-8') as file:
            file.write(json.dumps(payload, ensure_ascii=False))
            file.write('\n')

    def inbox(self, recipient: str) -> Tuple[MailMessage, ...]:
        return tuple(
            MailMessage(**row)
            for row in self._read_messages()
            if row.get('recipient') == recipient
        )

    def thread(self, task_id: str) -> Tuple[MailMessage, ...]:
        return tuple(
            MailMessage(**row)
            for row in self._read_messages()
            if row.get('task_id') == task_id
        )

    def summary_for(self, recipient: str, limit: int = 10) -> Dict[str, Any]:
        items = list(self.inbox(recipient))
        tail = items[-limit:]
        return {
            'recipient': recipient,
            'count': len(items),
            'subjects': [item.subject for item in tail],
            'task_ids': [item.task_id for item in tail if item.task_id],
        }

    def _read_messages(self) -> Tuple[Dict[str, Any], ...]:
        if not self._messages_file.exists():
            return tuple()
        rows = []
        with self._messages_file.open('r', encoding='utf-8') as file:
            for line in file:
                text = line.strip()
                if not text:
                    continue
                rows.append(json.loads(text))
        return tuple(rows)
