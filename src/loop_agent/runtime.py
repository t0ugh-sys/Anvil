from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .compression import CompressionConfig
from .core.serialization import run_result_to_dict
from .core.types import ContextSnapshot, ObserverFn, StopConfig
from .memory.jsonl_store import JsonlMemoryStore
from .permissions import PermissionManager
from .policies import Capability, ToolPolicy
from .run_recorder import RunRecorder
from .session import SessionStore
from .task_store import TaskStore


def build_jsonl_observer(path: str) -> ObserverFn:
    def observer(event: str, payload: Dict[str, Any]) -> None:
        record = {'event': event, 'payload': payload}
        with open(path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(record, ensure_ascii=False))
            file.write('\n')

    return observer


def merge_observers(observers: List[ObserverFn]) -> Optional[ObserverFn]:
    active = [item for item in observers if item is not None]
    if not active:
        return None

    def merged(event: str, payload: Dict[str, Any]) -> None:
        for observer in active:
            observer(event, payload)

    return merged


class CodeRuntime:
    def __init__(self, args, *, goal: str) -> None:
        self.args = args
        self.workspace_root = Path(args.workspace).resolve()
        self.goal = goal
        self.task_store = TaskStore((self.workspace_root / args.tasks_dir).resolve()) if args.tasks_dir else None
        self.compression_config = CompressionConfig(
            micro_keep_last_results=args.micro_compact_keep,
            max_context_tokens=args.max_context_tokens,
            recent_transcript_entries=args.recent_transcript_entries,
        )
        self.transcripts_dir = (self.workspace_root / args.transcripts_dir).resolve() if args.transcripts_dir else None
        self.run_id = args.run_id or self._default_run_id()
        self.permission_manager = PermissionManager(
            mode_name=str(getattr(args, 'permission_mode', 'balanced')),
        )

        sessions_dir_value = str(getattr(args, 'sessions_dir', '.loopagent/sessions'))
        sessions_root = Path(sessions_dir_value)
        if not sessions_root.is_absolute():
            if sessions_dir_value == '.loopagent/sessions':
                sessions_root = (self.workspace_root / sessions_root).resolve()
            else:
                sessions_root = sessions_root.resolve()
        if getattr(args, 'session_id', ''):
            self.session_store = SessionStore.load(root_dir=sessions_root, session_id=args.session_id)
            if not self.goal:
                self.goal = self.session_store.state.goal
            existing_cache = self.session_store.state.permission_cache
            self.permission_manager = PermissionManager(
                mode_name=str(getattr(args, 'permission_mode', 'balanced')),
                cache=existing_cache,
            )
            memory_dir_value = self.session_store.state.memory_run_dir or str(Path(args.memory_dir) / self.run_id)
            self.memory_run_dir = Path(memory_dir_value)
        else:
            self.memory_run_dir = Path(args.memory_dir) / self.run_id
            self.session_store = SessionStore.create(
                root_dir=sessions_root,
                workspace_root=self.workspace_root,
                goal=self.goal,
                memory_run_dir=self.memory_run_dir,
            )
        self.recorder: Optional[RunRecorder] = None
        self.memory_store = JsonlMemoryStore(memory_dir=self.memory_run_dir, summarize_every=args.summarize_every)
        self.observer = self._build_observer()

    def _default_run_id(self) -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

    def _build_observer(self) -> Optional[ObserverFn]:
        observers: List[ObserverFn] = []
        if self.args.observer_file:
            observers.append(build_jsonl_observer(self.args.observer_file))
        if self.args.record_run:
            self.recorder = RunRecorder.create(
                base_dir=Path(self.args.runs_dir),
                session_id=self.session_store.state.session_id,
            )
            self.observers_artifact_dir = str(self.recorder.run_dir)
            self.session_store.state.artifacts_dir = self.observers_artifact_dir
            self.session_store._write_session()
            observers.append(self.recorder.write_event)
        observers.append(self.memory_store.on_event)
        observers.append(self.session_store.append_event)
        return merge_observers(observers)

    def build_context_provider(self) -> Any:
        goal = self.goal
        memory_store = self.memory_store
        history_window = self.args.history_window

        def context_provider() -> ContextSnapshot:
            context = memory_store.load_context(goal=goal, last_k_steps=history_window)
            return ContextSnapshot(state_summary=context.state_summary, last_steps=context.last_steps)

        return context_provider

    def build_policy(self) -> ToolPolicy:
        return ToolPolicy(
            allowed=tuple(Capability),
            permission_manager=self.permission_manager,
        )

    def finalize(self, result) -> Dict[str, Any]:
        if self.observer is not None:
            self.observer(
                'run_finished',
                {'done': result.done, 'stop_reason': result.stop_reason.value, 'steps': result.steps},
            )
        else:
            self.memory_store.on_event(
                'run_finished',
                {'done': result.done, 'stop_reason': result.stop_reason.value, 'steps': result.steps},
            )
        memory_context = self.memory_store.load_context(goal=self.goal, last_k_steps=self.args.history_window)
        self.session_store.record_permission_cache(self.permission_manager.cache)
        payload = run_result_to_dict(result, include_history=self.args.include_history)
        payload['workspace'] = str(self.workspace_root)
        payload['provider'] = self.args.provider
        payload['model'] = self.args.model
        payload['memory_state'] = memory_context.state_summary
        payload['memory_last_steps'] = list(memory_context.last_steps)
        payload['memory_run_dir'] = str(self.memory_run_dir)
        payload['session_id'] = self.session_store.state.session_id
        payload['permission_mode'] = self.permission_manager.mode_name
        payload['permission_stats'] = dict(self.session_store.state.permission_stats)
        if self.recorder is not None:
            payload['run_dir'] = str(self.recorder.run_dir)
            self.recorder.write_summary(run_result_to_dict(result, include_history=True))
        self.session_store.write_summary(payload)
        return payload
