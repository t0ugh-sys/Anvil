from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


_VALID_TODO_STATUSES = {'pending', 'in_progress', 'completed'}


@dataclass(frozen=True)
class TodoItem:
    id: str
    content: str
    status: str = 'pending'

    def to_dict(self) -> Dict[str, str]:
        return {
            'id': self.id,
            'content': self.content,
            'status': self.status,
        }


@dataclass(frozen=True)
class TodoSnapshot:
    items: Tuple[TodoItem, ...] = tuple()
    rounds_since_update: int = 0


class TodoManager:
    def __init__(self, snapshot: TodoSnapshot | None = None) -> None:
        base = snapshot or TodoSnapshot()
        self._items: List[TodoItem] = list(base.items)
        self._updated = False

    def write(self, items: Iterable[Dict[str, Any]]) -> Tuple[TodoItem, ...]:
        parsed: List[TodoItem] = []
        in_progress_count = 0
        seen_ids = set()
        for raw in items:
            if not isinstance(raw, dict):
                raise ValueError('todo items must be objects')
            item_id = str(raw.get('id', '')).strip()
            content = str(raw.get('content', '')).strip()
            status = str(raw.get('status', 'pending')).strip() or 'pending'
            if not item_id:
                raise ValueError('todo item id is required')
            if item_id in seen_ids:
                raise ValueError(f'duplicate todo id: {item_id}')
            if not content:
                raise ValueError(f'todo content is required for: {item_id}')
            if status not in _VALID_TODO_STATUSES:
                raise ValueError(f'invalid todo status: {status}')
            if status == 'in_progress':
                in_progress_count += 1
            seen_ids.add(item_id)
            parsed.append(TodoItem(id=item_id, content=content, status=status))

        if in_progress_count > 1:
            raise ValueError('only one todo item can be in_progress')

        self._items = parsed
        self._updated = True
        return tuple(self._items)

    def snapshot(self, *, previous_rounds_since_update: int) -> TodoSnapshot:
        rounds_since_update = 0 if self._updated else previous_rounds_since_update + 1
        return TodoSnapshot(items=tuple(self._items), rounds_since_update=rounds_since_update)


def render_todo_lines(items: Iterable[TodoItem]) -> List[str]:
    lines: List[str] = []
    marker_map = {
        'pending': '[ ]',
        'in_progress': '[>]',
        'completed': '[x]',
    }
    for item in items:
        marker = marker_map.get(item.status, '[ ]')
        lines.append(f'{marker} {item.id}: {item.content}')
    return lines
