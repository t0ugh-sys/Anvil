from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .core.agent import AnvilAgent, RunResult, StepContext, StepResult
from .core.serialization import run_result_to_dict, run_result_to_json
from .core.types import StopConfig, StopReason
from .memory import JsonlMemoryStore, MemoryContext, MemoryStore

# Lazy-loaded submodules — imported on first access to reduce startup cost.
_LAZY_SUBMODULES: dict[str, str] = {
    'skills': '.skills',
    'config': '.config',
    'log': '.logging',
    'prompts': '.prompts',
    'errors': '.errors',
    'api': '.api',
    'task_graph': '.task_graph',
    'mailbox': '.mailbox',
    'subagents': '.subagents',
    'policies': '.policies',
    'worktree_manager': '.worktree_manager',
    'context_schema': '.context_schema',
    'scheduler': '.scheduler',
    'tool_use_loop': '.tool_use_loop',
    'ops': '.ops',
    'ui': '.ui',
    'todo': '.todo',
    'task_store': '.task_store',
    'compression': '.compression',
    'background': '.background',
    'permissions': '.permissions',
    'runtime': '.runtime',
    'session': '.session',
    'tool_spec': '.tool_spec',
    'commands': '.commands',
    'services': '.services',
    'entrypoints': '.entrypoints',
    'team_runtime': '.team_runtime',
}

if TYPE_CHECKING:
    from . import (  # noqa: F401
        skills as skills,
        config as config,
        logging as log,
        prompts as prompts,
        errors as errors,
        api as api,
        task_graph as task_graph,
        mailbox as mailbox,
        subagents as subagents,
        policies as policies,
        worktree_manager as worktree_manager,
        context_schema as context_schema,
        scheduler as scheduler,
        tool_use_loop as tool_use_loop,
        ops as ops,
        ui as ui,
        todo as todo,
        task_store as task_store,
        compression as compression,
        background as background,
        permissions as permissions,
        runtime as runtime,
        session as session,
        tool_spec as tool_spec,
        commands as commands,
        services as services,
        entrypoints as entrypoints,
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
    # Lazy-loaded submodules
    'skills',
    'config',
    'log',
    'prompts',
    'errors',
    'api',
    'task_graph',
    'mailbox',
    'subagents',
    'policies',
    'worktree_manager',
    'context_schema',
    'scheduler',
    'tool_use_loop',
    'ops',
    'ui',
    'todo',
    'task_store',
    'compression',
    'background',
    'permissions',
    'runtime',
    'session',
    'tool_spec',
    'commands',
    'services',
    'entrypoints',
    'team_runtime',
]
