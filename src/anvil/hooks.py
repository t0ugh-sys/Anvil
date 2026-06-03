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
import shlex
import subprocess
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Sequence

__all__ = [
    'HookEvent',
    'HookResult',
    'HookConfig',
    'HookManager',
    'HookInput',
    'HookOutput',
    'run_hook',
    'run_hooks_for_event',
    'build_hook_input_for_tool',
]


def _event_name(event: HookEvent | str) -> str:
    """Extract event name string from HookEvent or plain string."""
    return event.value if isinstance(event, HookEvent) else event


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
    shell: bool = False


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
        # Default to shlex.split for safety; use shell=True only when explicitly requested.
        cmd: str | list[str] = (
            config.command if config.shell else shlex.split(config.command)
        )
        result = subprocess.run(
            cmd,
            shell=config.shell,
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
        event_name = _event_name(event)
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
        event_name = _event_name(event)
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
        event_name = _event_name(event)
        with self._lock:
            return bool(self._hooks.get(event_name))

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
        event=_event_name(event),
        tool_name=tool_name,
        tool_input=tool_input,
        tool_output=tool_output,
        session_id=session_id,
        workspace_root=workspace_root,
    )


# ============== Security Monitor ==============
# Based on Zero2Agent article #41: Multi-Agent Debugging & Monitoring
# Tracks tool usage patterns and detects anomalous behavior.


import time as _time


class SecurityEvent:
    """A security-related event for audit logging."""

    __slots__ = ('timestamp', 'event_type', 'severity', 'tool_name', 'details')

    def __init__(
        self,
        event_type: str,
        severity: str,
        tool_name: str = '',
        details: str = '',
    ) -> None:
        self.timestamp = _time.time()
        self.event_type = event_type
        self.severity = severity  # 'info', 'warning', 'critical'
        self.tool_name = tool_name
        self.details = details

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'event_type': self.event_type,
            'severity': self.severity,
            'tool_name': self.tool_name,
            'details': self.details[:500],
        }


class SecurityMonitor:
    """Monitors tool usage patterns and detects anomalies.

    Tracks per-tool call counts within a sliding time window.
    Alerts when a tool exceeds the configured call threshold.

    Usage::

        monitor = SecurityMonitor(window_seconds=60, max_calls_per_tool=20)
        alert = monitor.record_call('run_command')
        if alert:
            # Alert string describes the anomaly
            log_warning(alert)
    """

    def __init__(
        self,
        window_seconds: float = 60.0,
        max_calls_per_tool: int = 30,
    ) -> None:
        self._window = window_seconds
        self._max_calls = max_calls_per_tool
        self._call_times: Dict[str, list] = {}  # tool_name -> [timestamps]
        self._events: list[SecurityEvent] = []
        self._blocked_tools: set = set()

    def record_call(self, tool_name: str) -> str | None:
        """Record a tool call and check for anomalies.

        Returns an alert string if an anomaly is detected, else None.
        """
        now = _time.time()
        cutoff = now - self._window

        # Initialize tracking
        if tool_name not in self._call_times:
            self._call_times[tool_name] = []

        # Prune old timestamps
        self._call_times[tool_name] = [
            t for t in self._call_times[tool_name] if t > cutoff
        ]

        # Record this call
        self._call_times[tool_name].append(now)
        count = len(self._call_times[tool_name])

        # Check threshold
        if count >= self._max_calls:
            self._events.append(SecurityEvent(
                event_type='tool_rate_exceeded',
                severity='warning',
                tool_name=tool_name,
                details=f'{tool_name} called {count} times in {self._window:.0f}s (limit: {self._max_calls})',
            ))
            return (
                f'SECURITY WARNING: {tool_name} called {count} times in '
                f'{self._window:.0f}s. Possible loop or abuse.'
            )
        return None

    def block_tool(self, tool_name: str) -> None:
        """Block a tool from being called."""
        self._blocked_tools.add(tool_name)
        self._events.append(SecurityEvent(
            event_type='tool_blocked',
            severity='critical',
            tool_name=tool_name,
            details=f'{tool_name} manually blocked',
        ))

    def unblock_tool(self, tool_name: str) -> None:
        """Unblock a tool."""
        self._blocked_tools.discard(tool_name)

    def is_blocked(self, tool_name: str) -> bool:
        """Check if a tool is blocked."""
        return tool_name in self._blocked_tools

    def get_call_count(self, tool_name: str) -> int:
        """Get current call count for a tool within the window."""
        now = _time.time()
        cutoff = now - self._window
        times = self._call_times.get(tool_name, [])
        return sum(1 for t in times if t > cutoff)

    def get_events(self, severity: str = '') -> list[Dict[str, Any]]:
        """Get recorded security events, optionally filtered by severity."""
        events = self._events
        if severity:
            events = [e for e in events if e.severity == severity]
        return [e.to_dict() for e in events]

    def reset(self) -> None:
        """Reset all tracking state."""
        self._call_times.clear()
        self._events.clear()
        self._blocked_tools.clear()

    def summary(self) -> Dict[str, Any]:
        """Get a summary of security monitoring state."""
        tool_counts = {
            name: self.get_call_count(name)
            for name in self._call_times
        }
        return {
            'window_seconds': self._window,
            'max_calls_per_tool': self._max_calls,
            'active_tools': tool_counts,
            'blocked_tools': list(self._blocked_tools),
            'total_events': len(self._events),
            'critical_events': len([e for e in self._events if e.severity == 'critical']),
        }
