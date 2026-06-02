from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Mapping, Tuple

if TYPE_CHECKING:
    from .permissions import PermissionManager


class Capability(str, Enum):
    read = 'read'
    write = 'write'
    execute = 'execute'
    network = 'network'
    memory = 'memory'


__all__ = ['TOOL_CAPABILITIES', 'PolicyManager', 'LoopDetector', 'TokenBudget']


import hashlib
import json as _json
from collections import deque
from typing import List, Optional


class LoopDetector:
    """Detects when the agent repeats the same tool+args combination.

    Tracks a sliding window of (tool_name, normalized_args_hash) tuples.
    When the same combination appears ``max_repeats`` times in the window,
    :meth:`check` returns an advisory string; otherwise ``None``.
    """

    def __init__(self, max_repeats: int = 3, window: int = 6) -> None:
        self.max_repeats = max_repeats
        self._window = window
        self._history: deque = deque(maxlen=window)

    def _key(self, tool_name: str, tool_args: dict) -> str:
        raw = _json.dumps({'n': tool_name, 'a': tool_args}, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()

    def check(self, tool_name: str, tool_args: dict) -> Optional[str]:
        """Record a tool invocation and check for loops.

        Returns an advisory message if a loop is detected, else ``None``.
        """
        key = self._key(tool_name, tool_args)
        self._history.append(key)
        if len(self._history) >= self.max_repeats:
            recent = list(self._history)[-self.max_repeats :]
            if len(set(recent)) == 1:
                return (
                    f'LOOP DETECTED: {tool_name} repeated {self.max_repeats} times '
                    f'with identical arguments. You MUST try a fundamentally different approach.'
                )
        return None

    def reset(self) -> None:
        self._history.clear()


class TokenBudget:
    """Tracks cumulative token usage and enforces a hard ceiling.

    Call :meth:`record` after each LLM round with the token counts.
    Call :meth:`check` before each LLM round; it returns an advisory
    string when the budget is exceeded, else ``None``.
    """

    def __init__(
        self,
        max_input_tokens: int = 0,
        max_output_tokens: int = 0,
        max_total_tokens: int = 0,
    ) -> None:
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.max_total_tokens = max_total_tokens
        self._input_tokens = 0
        self._output_tokens = 0

    def record(self, input_tokens: int, output_tokens: int) -> None:
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens

    @property
    def total_tokens(self) -> int:
        return self._input_tokens + self._output_tokens

    def check(self) -> Optional[str]:
        """Return advisory string if budget exceeded, else ``None``."""
        if self.max_total_tokens and self.total_tokens > self.max_total_tokens:
            return (
                f'TOKEN BUDGET EXCEEDED: used {self.total_tokens:,} / '
                f'{self.max_total_tokens:,} total tokens. Stop and produce final output.'
            )
        if self.max_input_tokens and self._input_tokens > self.max_input_tokens:
            return (
                f'INPUT TOKEN BUDGET EXCEEDED: used {self._input_tokens:,} / '
                f'{self.max_input_tokens:,} input tokens.'
            )
        if self.max_output_tokens and self._output_tokens > self.max_output_tokens:
            return (
                f'OUTPUT TOKEN BUDGET EXCEEDED: used {self._output_tokens:,} / '
                f'{self.max_output_tokens:,} output tokens.'
            )
        return None

    def to_dict(self) -> dict:
        return {
            'input_tokens': self._input_tokens,
            'output_tokens': self._output_tokens,
            'total_tokens': self.total_tokens,
            'max_total_tokens': self.max_total_tokens,
        }


TOOL_CAPABILITIES: Dict[str, Tuple[Capability, ...]] = {
    'read_file': (Capability.read,),
    'search': (Capability.read,),
    'load_skill': (Capability.read,),
    'write_file': (Capability.write,),
    'apply_patch': (Capability.write,),
    'todo_write': (Capability.memory,),
    'compact': (Capability.memory,),
    'run_command': (Capability.execute,),
    'run_command_async': (Capability.execute,),
    'web_search': (Capability.network,),
    'fetch_url': (Capability.network,),
    'analyze_memory': (Capability.memory, Capability.read),
    'git_status': (Capability.execute, Capability.read),
    'git_branch_list': (Capability.execute, Capability.read),
    'git_checkout': (Capability.execute, Capability.write),
    'git_pull': (Capability.execute, Capability.network),
    'git_merge': (Capability.execute, Capability.write),
    'git_merge_and_push': (Capability.execute, Capability.write, Capability.network),
    'git_push': (Capability.execute, Capability.network),
    'gh_auth_status': (Capability.execute, Capability.network),
    'gh_repo_list': (Capability.execute, Capability.network),
    'gh_repo_create': (Capability.execute, Capability.network),
    'gh_repo_clone': (Capability.execute, Capability.network, Capability.write),
    'gh_issue_list': (Capability.execute, Capability.network),
    'gh_issue_create': (Capability.execute, Capability.network, Capability.write),
    'gh_issue_close': (Capability.execute, Capability.network, Capability.write),
    'gh_pr_list': (Capability.execute, Capability.network),
    'gh_pr_create': (Capability.execute, Capability.network, Capability.write),
    'gh_pr_view': (Capability.execute, Capability.network),
    'gh_pr_checks': (Capability.execute, Capability.network),
    'gh_pr_comment': (Capability.execute, Capability.network, Capability.write),
    'gh_pr_merge': (Capability.execute, Capability.network, Capability.write),
}


@dataclass(frozen=True)
class ToolPolicy:
    allowed: Tuple[Capability, ...] = field(default_factory=tuple)
    denied: Tuple[Capability, ...] = field(default_factory=tuple)
    permission_manager: PermissionManager | None = None

    @classmethod
    def allow_all(cls) -> 'ToolPolicy':
        return cls(allowed=tuple(Capability))

    @classmethod
    def read_only(cls) -> 'ToolPolicy':
        return cls(allowed=(Capability.read, Capability.memory))

    def allows_tool(self, tool_name: str) -> bool:
        required = TOOL_CAPABILITIES.get(tool_name, tuple())
        if not required:
            return True
        if any(capability in self.denied for capability in required):
            return False
        return all(capability in self.allowed for capability in required)

    def denied_capabilities_for_tool(self, tool_name: str) -> Tuple[Capability, ...]:
        required = TOOL_CAPABILITIES.get(tool_name, tuple())
        blocked = [capability for capability in required if capability in self.denied or capability not in self.allowed]
        return tuple(blocked)

    def to_dict(self) -> Dict[str, object]:
        return {
            'allowed': [capability.value for capability in self.allowed],
            'denied': [capability.value for capability in self.denied],
        }


def policy_from_name(name: str) -> ToolPolicy:
    normalized = name.strip().lower()
    if normalized in {'full', 'allow_all'}:
        return ToolPolicy.allow_all()
    if normalized in {'read_only', 'readonly'}:
        return ToolPolicy.read_only()
    raise ValueError(f'unknown policy preset: {name}')


def build_tool_permissions() -> Mapping[str, Tuple[Capability, ...]]:
    return dict(TOOL_CAPABILITIES)
