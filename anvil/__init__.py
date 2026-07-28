from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .core.agent import AnvilAgent, RunResult, StepContext, StepResult
from .core.serialization import run_result_to_dict, run_result_to_json
from .core.types import StopConfig, StopReason
from .memory import JsonlMemoryStore, MemoryContext, MemoryStore

# Lazy-loaded submodules — imported on first access to reduce startup cost.
_LAZY_SUBMODULES: dict[str, str] = {
    # Subpackages
    'infra': '.infra',
    'agent': '.agent',
    'config': '.config',
    'compression': '.compression',
    # Top-level modules
    'prompts': '.prompts',
    'errors': '.errors',
    'api': '.api',
    'worktree_manager': '.worktree_manager',
    'ops': '.ops',
    'ui': '.ui',
    'todo': '.todo',
    'tool_spec': '.tool_spec',
    'commands': '.commands',
    'services': '.services',
    'entrypoints': '.entrypoints',
    # Legacy aliases for backward compat (point to canonical paths)
    'skills': '.infra.skills',
    'log': '.infra.logging',
    'policies': '.infra.policies',
    'permissions': '.infra.permissions',
    'hooks': '.infra.hooks',
    'retry': '.infra.retry',
    'subagents': '.agent.subagents',
    'background': '.agent.background',
    'tool_use_loop': '.agent.loop',
    'runtime': '.runtime.code',
    'session': '.runtime.session',
    'mailbox': '.runtime.mailbox',
    'scheduler': '.runtime.scheduler',
    'task_graph': '.runtime.task_graph',
    'task_store': '.runtime.task_store',
    'team_runtime': '.runtime.team',
}

if TYPE_CHECKING:
    from . import (  # noqa: F401
        # Subpackages
        infra as infra,
        agent as agent,
        config as config,
        compression as compression,
        # Top-level modules
        prompts as prompts,
        errors as errors,
        api as api,
        worktree_manager as worktree_manager,
        ops as ops,
        ui as ui,
        todo as todo,
        tool_spec as tool_spec,
        commands as commands,
        services as services,
        entrypoints as entrypoints,
        # Legacy aliases (backward compat)
        skills as skills,
        log as log,
        policies as policies,
        permissions as permissions,
        hooks as hooks,
        retry as retry,
        subagents as subagents,
        background as background,
        tool_use_loop as tool_use_loop,
        runtime as runtime,
        session as session,
        mailbox as mailbox,
        scheduler as scheduler,
        task_graph as task_graph,
        task_store as task_store,
        team_runtime as team_runtime,
    )


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(_LAZY_SUBMODULES[name], __name__)
        globals()[name] = module
        return module
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__() -> list[str]:
    return list(globals().keys()) + list(_LAZY_SUBMODULES.keys())


__all__ = [
    'AnvilAgent',
    'RunResult',
    'StepContext',
    'StepResult',
    'StopConfig',
    'StopReason',
    'run_result_to_dict',
    'run_result_to_json',
    'MemoryStore',
    'MemoryContext',
    'JsonlMemoryStore',
    # Subpackages
    'infra',
    'agent',
    'config',
    'compression',
    # Top-level modules
    'prompts',
    'errors',
    'api',
    'worktree_manager',
    'ops',
    'ui',
    'todo',
    'tool_spec',
    'commands',
    'services',
    'entrypoints',
    # Legacy aliases (backward compat)
    'skills',
    'log',
    'policies',
    'permissions',
    'hooks',
    'retry',
    'subagents',
    'background',
    'tool_use_loop',
    'runtime',
    'session',
    'mailbox',
    'scheduler',
    'task_graph',
    'task_store',
    'team_runtime',
]
