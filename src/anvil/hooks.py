"""
Hook system for lifecycle events.

Inspired by Claude Code's hooks: user-defined shell commands triggered at
lifecycle points. Hooks receive structured JSON via stdin and return JSON output.

Supported events:
- PreToolUse: Before tool execution (can approve/block/modify)
- PostToolUse: After tool execution (can inject follow-up context)
- SessionStart: When a session begins
- Stop: When the agent is about to stop
- PreCompact: Before context compaction

Hook configuration (in settings):
{
    "hooks": {
        "PreToolUse": [
            {"command": "python validate.py", "timeout_s": 10}
        ],
        "PostToolUse": [
            {"command": "python log_result.py"}
        ]
    }
}
"""

from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence


class HookEvent(str, Enum):
    """Lifecycle events that trigger hooks."""
    PreToolUse = 'PreToolUse'
    PostToolUse = 'PostToolUse'
    SessionStart = 'SessionStart'
    Stop = 'Stop'
    PreCompact = 'PreCompact'


@dataclass(frozen=True)
class HookConfig:
    """Configuration for a single hook."""
    command: str
    timeout_s: float = 30.0
    async_mode: bool = False


@dataclass(frozen=True)
class HookInput:
    """Structured input passed to hook via stdin."""
    event: str
    tool_name: str = ''
    tool_input: Dict[str, Any] = field(default_factory=dict)
    tool_output: str = ''
    session_id: str = ''
    workspace_root: str = ''
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            'event': self.event,
            'tool_name': self.tool_name,
            'tool_input': self.tool_input,
            'tool_output': self.tool_output,
            'session_id': self.session_id,
            'workspace_root': self.workspace_root,
            'metadata': self.metadata,
        }, ensure_ascii=False)


@dataclass(frozen=True)
class HookOutput:
    """Structured output from hook via stdout."""
    approve: bool = True
    modified_input: Dict[str, Any] | None = None
    context: str = ''
    error: str = ''

    @classmethod
    def from_json(cls, data: str) -> HookOutput:
        try:
            parsed = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return cls(approve=True)
        return cls(
            approve=parsed.get('approve', True),
            modified_input=parsed.get('modified_input'),
            context=parsed.get('context', ''),
            error=parsed.get('error', ''),
        )

    @classmethod
    def approved(cls) -> HookOutput:
        return cls(approve=True)

    @classmethod
    def blocked(cls, reason: str) -> HookOutput:
        return cls(approve=False, error=reason)


@dataclass
class HookResult:
    """Result of running hooks for an event."""
    approved: bool = True
    modified_input: Dict[str, Any] | None = None
    context: str = ''
    error: str = ''
    hooks_run: int = 0


def run_hook(
    config: HookConfig,
    hook_input: HookInput,
    *,
    timeout_s: float | None = None,
) -> HookOutput:
    """Run a single hook command.

    Sends hook_input as JSON via stdin, reads JSON output from stdout.
    Returns HookOutput with approval decision and optional modifications.
    """
    timeout = timeout_s or config.timeout_s
    try:
        result = subprocess.run(
            config.command,
            shell=True,
            input=hook_input.to_json(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return HookOutput.blocked(f'hook exited with code {result.returncode}: {result.stderr[:200]}')
        return HookOutput.from_json(result.stdout)
    except subprocess.TimeoutExpired:
        return HookOutput.blocked(f'hook timed out after {timeout}s')
    except Exception as exc:
        return HookOutput.blocked(f'hook error: {str(exc)[:200]}')


def run_hooks_for_event(
    event: HookEvent,
    hooks: Sequence[HookConfig],
    hook_input: HookInput,
) -> HookResult:
    """Run all hooks for a given event.

    Returns combined result. If any hook blocks, the result is blocked.
    Context strings from all hooks are concatenated.
    """
    result = HookResult(approved=True)

    for config in hooks:
        if config.async_mode:
            # Async hooks run in background, don't block
            thread = threading.Thread(
                target=run_hook,
                args=(config, hook_input),
                daemon=True,
            )
            thread.start()
            result.hooks_run += 1
            continue

        output = run_hook(config, hook_input)
        result.hooks_run += 1

        if not output.approve:
            result.approved = False
            result.error = output.error
            return result

        if output.modified_input is not None:
            result.modified_input = output.modified_input

        if output.context:
            result.context = (result.context + '\n' + output.context).strip()

    return result


class HookManager:
    """Manages hook registration and execution.

    Thread-safe: hooks can be registered and executed from any thread.
    """

    def __init__(self) -> None:
        self._hooks: Dict[str, List[HookConfig]] = {}
        self._lock = threading.Lock()

    def register(self, event: HookEvent | str, config: HookConfig) -> None:
        """Register a hook for an event."""
        event_name = event.value if isinstance(event, HookEvent) else event
        with self._lock:
            if event_name not in self._hooks:
                self._hooks[event_name] = []
            self._hooks[event_name].append(config)

    def register_command(
        self,
        event: HookEvent | str,
        command: str,
        *,
        timeout_s: float = 30.0,
        async_mode: bool = False,
    ) -> None:
        """Register a shell command hook for an event."""
        self.register(event, HookConfig(command=command, timeout_s=timeout_s, async_mode=async_mode))

    def get_hooks(self, event: HookEvent | str) -> List[HookConfig]:
        """Get all hooks registered for an event."""
        event_name = event.value if isinstance(event, HookEvent) else event
        with self._lock:
            return list(self._hooks.get(event_name, []))

    def run_event(
        self,
        event: HookEvent,
        hook_input: HookInput,
    ) -> HookResult:
        """Run all hooks for an event."""
        hooks = self.get_hooks(event)
        if not hooks:
            return HookResult(approved=True, hooks_run=0)
        return run_hooks_for_event(event, hooks, hook_input)

    def has_hooks(self, event: HookEvent | str) -> bool:
        """Check if any hooks are registered for an event."""
        return len(self.get_hooks(event)) > 0

    def clear(self) -> None:
        """Remove all registered hooks."""
        with self._lock:
            self._hooks.clear()


def build_hook_input_for_tool(
    event: HookEvent,
    tool_name: str,
    tool_input: Dict[str, Any],
    *,
    tool_output: str = '',
    session_id: str = '',
    workspace_root: str = '',
) -> HookInput:
    """Build HookInput for a tool-related event."""
    return HookInput(
        event=event.value if isinstance(event, HookEvent) else event,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_output=tool_output,
        session_id=session_id,
        workspace_root=workspace_root,
    )
