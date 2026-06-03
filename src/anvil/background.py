from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Dict, List, Tuple

from .agent_protocol import ToolResult

__all__ = ['BackgroundCommandRunner', 'BackgroundTaskInfo']


@dataclass(frozen=True)
class BackgroundTaskInfo:
    id: str
    command: Tuple[str, ...]
    status: str

    def to_dict(self) -> dict[str, object]:
        return {
            'id': self.id,
            'command': list(self.command),
            'status': self.status,
        }


class BackgroundCommandRunner:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self._lock = threading.Lock()
        self._counter = 0
        self._tasks: Dict[str, BackgroundTaskInfo] = {}
        self._notifications: Queue[ToolResult] = Queue()
        self._processes: Dict[str, subprocess.Popen] = {}
        self._output_buffers: Dict[str, List[str]] = {}

    def spawn(self, *, command: List[str], call_id: str) -> ToolResult:
        normalized = [str(item) for item in command if str(item)]
        if not normalized:
            return ToolResult(id=call_id, ok=False, output='', error='cmd list is required')

        with self._lock:
            self._counter += 1
            task_id = f'bg_{self._counter}'
            self._tasks[task_id] = BackgroundTaskInfo(id=task_id, command=tuple(normalized), status='running')

        thread = threading.Thread(
            target=self._run_task,
            args=(task_id, call_id, normalized),
            daemon=True,
        )
        thread.start()
        return ToolResult(
            id=call_id,
            ok=True,
            output=f'background task started: {task_id}',
            error=None,
        )

    def _run_task(self, task_id: str, call_id: str, command: List[str]) -> None:
        try:
            proc = subprocess.Popen(
                command,
                cwd=str(self.workspace_root),
                shell=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding='utf-8',
                errors='replace',
            )
            with self._lock:
                self._processes[task_id] = proc
                self._output_buffers[task_id] = []
            # Stream output line by line
            assert proc.stdout is not None
            for line in proc.stdout:
                with self._lock:
                    buf = self._output_buffers.get(task_id)
                    if buf is not None:
                        buf.append(line)
            proc.wait()
            merged = ''.join(self._output_buffers.get(task_id, []))
            ok = proc.returncode == 0
            result = ToolResult(
                id=call_id,
                ok=ok,
                output=f'background[{task_id}] {merged.strip()}'.strip(),
                error=None if ok else f'exit={proc.returncode}',
            )
            status = 'completed' if ok else 'failed'
        except FileNotFoundError as exc:
            result = ToolResult(id=call_id, ok=False, output=f'background[{task_id}]', error=f'command not found: {exc.filename}')
            status = 'failed'
        except Exception as exc:
            result = ToolResult(id=call_id, ok=False, output=f'background[{task_id}]', error=str(exc))
            status = 'failed'

        with self._lock:
            info = self._tasks.get(task_id)
            if info is not None:
                self._tasks[task_id] = BackgroundTaskInfo(id=task_id, command=info.command, status=status)
            # Release process handle and buffer after completion to avoid leaks
            self._processes.pop(task_id, None)
            self._output_buffers.pop(task_id, None)
        self._notifications.put(result)

    def drain_notifications(self) -> Tuple[ToolResult, ...]:
        import queue
        results: List[ToolResult] = []
        while True:
            try:
                result = self._notifications.get_nowait()
            except queue.Empty:
                break
            results.append(result)
        return tuple(results)

    def snapshot(self) -> Tuple[BackgroundTaskInfo, ...]:
        with self._lock:
            return tuple(self._tasks.values())

    def read_output(self, task_id: str, tail: int = 50) -> str:
        """Read the latest output lines from a running or completed task."""
        with self._lock:
            buf = self._output_buffers.get(task_id)
            if buf is None:
                return ''
            return ''.join(buf[-tail:])

    def kill_task(self, task_id: str) -> bool:
        """Kill a running background process. Returns True if killed."""
        with self._lock:
            proc = self._processes.get(task_id)
            if proc is None:
                return False
            try:
                proc.terminate()
                return True
            except OSError:
                return False
