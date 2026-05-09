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
        denied = set(self.denied)
        allowed = set(self.allowed)
        if any(capability in denied for capability in required):
            return False
        return all(capability in allowed for capability in required)

    def denied_capabilities_for_tool(self, tool_name: str) -> Tuple[Capability, ...]:
        required = TOOL_CAPABILITIES.get(tool_name, tuple())
        denied = set(self.denied)
        allowed = set(self.allowed)
        blocked = [capability for capability in required if capability in denied or capability not in allowed]
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
