from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "run-schema-v1"  # keep in sync with loop_agent.run_schema.SCHEMA_VERSION
from .base import MemoryContext


def _default_summary() -> dict[str, Any]:
    return {
        'schema_version': SCHEMA_VERSION,
        'goal': '',
        'current_plan': [],
        'facts': [],
        'work_done': [],
        'open_questions': [],
        'next_actions': [],
        'steps': 0,
    }


@dataclass
class JsonlMemoryStore:
    memory_dir: Path
    summarize_every: int = 5
    _events_file: Path = field(init=False)
    _state_file: Path = field(init=False)
    _summary_file: Path = field(init=False)

    def __post_init__(self) -> None:
        if self.summarize_every < 1:
            raise ValueError('summarize_every must be >= 1')
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._events_file = self.memory_dir / 'events.jsonl'
        self._state_file = self.memory_dir / 'state.json'
        self._summary_file = self.memory_dir / 'summary.json'
        if not self._state_file.exists():
            self._write_state(
                {
                    'schema_version': SCHEMA_VERSION,
                    'goal': '',
                    'step_index': 0,
                    'last_output': '',
                    'history_tail': [],
                }
            )
        if not self._summary_file.exists():
            self._write_summary(_default_summary())

    def on_event(self, event: str, payload: dict[str, Any]) -> None:
        row = {'event': event, 'payload': payload}
        with self._events_file.open('a', encoding='utf-8') as file:
            file.write(json.dumps(row, ensure_ascii=False))
            file.write('\n')
        self._update_state(event, payload)
        event_count = self._count_events()
        if (event_count % self.summarize_every) == 0 or event == 'run_finished':
            self._summarize()

    def load_context(self, *, goal: str, last_k_steps: int) -> MemoryContext:
        state = self._read_summary()
        if not state.get('goal'):
            state['goal'] = goal
        last_steps = self._read_last_steps(last_k_steps)
        return MemoryContext(state_summary=state, last_steps=tuple(last_steps))

    def append_event(self, event: str, payload: dict[str, Any]) -> None:
        self.on_event(event, payload)

    def get_context(self, *, last_k_steps: int) -> MemoryContext:
        return self.load_context(goal='', last_k_steps=last_k_steps)

    def _count_events(self) -> int:
        if not self._events_file.exists():
            return 0
        with self._events_file.open('r', encoding='utf-8') as file:
            return sum(1 for _ in file)

    def _read_state(self) -> dict[str, Any]:
        default_state: dict[str, Any] = {
            'schema_version': SCHEMA_VERSION,
            'goal': '',
            'step_index': 0,
            'last_output': '',
            'history_tail': [],
        }
        if not self._state_file.exists():
            return default_state
        try:
            state = json.loads(self._state_file.read_text(encoding='utf-8'))
            if isinstance(state, dict) and 'schema_version' not in state:
                state['schema_version'] = SCHEMA_VERSION
            return state if isinstance(state, dict) else default_state
        except json.JSONDecodeError:
            return default_state

    def _write_state(self, state: dict[str, Any]) -> None:
        self._state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')

    def _read_summary(self) -> dict[str, Any]:
        if not self._summary_file.exists():
            return _default_summary()
        try:
            state = json.loads(self._summary_file.read_text(encoding='utf-8'))
            if isinstance(state, dict) and 'schema_version' not in state:
                state['schema_version'] = SCHEMA_VERSION
            return state if isinstance(state, dict) else _default_summary()
        except json.JSONDecodeError:
            return _default_summary()

    def _write_summary(self, state: dict[str, Any]) -> None:
        self._summary_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')

    def _read_events(self) -> list[dict[str, Any]]:
        if not self._events_file.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self._events_file.open('r', encoding='utf-8') as file:
            for line in file:
                text = line.strip()
                if not text:
                    continue
                try:
                    rows.append(json.loads(text))
                except json.JSONDecodeError:
                    continue
        return rows

    def _read_last_steps(self, last_k_steps: int) -> list[str]:
        if last_k_steps <= 0:
            return []
        rows = self._read_events()
        steps: list[str] = []
        for row in rows:
            if row.get('event') == 'step_succeeded':
                payload = row.get('payload', {})
                if isinstance(payload, dict):
                    output = payload.get('output', '')
                    if isinstance(output, str) and output:
                        steps.append(output)
        return steps[-last_k_steps:]

    def _summarize(self) -> None:
        rows = self._read_events()
        state = _default_summary()
        for row in rows:
            event = row.get('event')
            payload = row.get('payload', {})
            if not isinstance(payload, dict):
                continue

            if event == 'run_started':
                goal = payload.get('goal', '')
                if isinstance(goal, str):
                    state['goal'] = goal
                facts = payload.get('facts', [])
                if isinstance(facts, list):
                    state['facts'] = [item for item in facts if isinstance(item, str)]

            if event == 'step_started':
                state['steps'] = int(payload.get('step', state['steps'])) + 1
                plan = payload.get('plan', [])
                if isinstance(plan, list):
                    state['current_plan'] = [item for item in plan if isinstance(item, str)]

            if event == 'step_succeeded':
                output = payload.get('output', '')
                if isinstance(output, str) and output:
                    state['work_done'].append(output)
                metadata = payload.get('metadata', {})
                if isinstance(metadata, dict):
                    plan = metadata.get('plan', [])
                    if isinstance(plan, list):
                        state['current_plan'] = [item for item in plan if isinstance(item, str)]
                    tool_results = metadata.get('tool_results', [])
                    if isinstance(tool_results, list):
                        for item in tool_results:
                            if not isinstance(item, dict):
                                continue
                            tool_id = item.get('id')
                            ok = item.get('ok')
                            if isinstance(tool_id, str) and isinstance(ok, bool):
                                state['work_done'].append(f'tool[{tool_id}] {"ok" if ok else "failed"}')

            if event == 'step_failed':
                error = payload.get('error', '')
                if isinstance(error, str) and error:
                    state['open_questions'].append(error)

        work_done = state['work_done'][-20:]
        open_questions = state['open_questions'][-20:]
        state['work_done'] = work_done
        state['open_questions'] = open_questions
        state['next_actions'] = state['current_plan'][:3]
        self._write_summary(state)

    def summarize_now(self) -> None:
        self._summarize()

    def _update_state(self, event: str, payload: dict[str, Any]) -> None:
        state = self._read_state()
        goal = payload.get('goal')
        if event == 'run_started' and isinstance(goal, str):
            state['goal'] = goal
        step = payload.get('step')
        if isinstance(step, int):
            state['step_index'] = step
        output = payload.get('output')
        if isinstance(output, str) and output:
            state['last_output'] = output
            history_tail = state.get('history_tail', [])
            if not isinstance(history_tail, list):
                history_tail = []
            history_tail.append(output)
            state['history_tail'] = history_tail[-20:]
        self._write_state(state)
