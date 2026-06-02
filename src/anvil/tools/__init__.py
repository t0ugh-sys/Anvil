"""Modular tool system for Anvil.

This package provides a clean, modular architecture for tools inspired by
Claude Code's tool system. Each tool category is in its own module:

- file_tools: read_file, write_file, apply_patch
- search_tools: search, web_search, fetch_url
- command_tools: run_command, run_command_async
- memory_tools: analyze_memory, todo_write, load_skill, compact

The main entry point is build_default_tools() which returns a dispatch map.
"""
from __future__ import annotations

from typing import Iterable

from ..agent_protocol import ToolResult
from ..policies import TOOL_CAPABILITIES
from ..tool_spec import ToolRisk, ToolSpec
from .base import (
    ToolContext,
    ToolDispatchMap,
    ToolFn,
    ToolRegistration,
    iter_searchable_files,
    resolve_inside_workspace,
)
from .command_tools import command_tool_specs, run_command_async_tool, run_command_tool
from .file_tools import apply_patch_tool, file_tool_specs, read_file_tool, write_file_tool
from .memory_tools import (
    analyze_memory_tool,
    compact_tool,
    load_skill_tool,
    memory_tool_specs,
    todo_write_tool,
)
from .search_tools import fetch_url_tool, search_tool, search_tool_specs, web_search_tool

# Re-export for backward compatibility
__all__ = [
    # Types
    'ToolContext',
    'ToolDispatchMap',
    'ToolFn',
    'ToolRegistration',
    # Functions
    'build_default_tools',
    'execute_tool_call',
    'register_tool_handler',
    'builtin_tool_registrations',
    'builtin_tool_specs',
    # Individual tools
    'read_file_tool',
    'write_file_tool',
    'apply_patch_tool',
    'search_tool',
    'web_search_tool',
    'fetch_url_tool',
    'run_command_tool',
    'run_command_async_tool',
    'analyze_memory_tool',
    'todo_write_tool',
    'load_skill_tool',
    'compact_tool',
    # Utilities
    'iter_searchable_files',
    'resolve_inside_workspace',
]

def _build_tool_dispatch_map(registrations: Iterable[ToolRegistration]) -> ToolDispatchMap:
    """Build a dispatch map from registrations."""
    return {name: handler for name, handler in registrations}


def register_tool_handler(dispatch_map: ToolDispatchMap, name: str, handler: ToolFn) -> ToolDispatchMap:
    """Register a tool handler in the dispatch map."""
    dispatch_map[name] = handler
    return dispatch_map


def _builtin_core_tool_registrations() -> list[ToolRegistration]:
    """Core tool registrations (file, search, command, memory)."""
    return [
        ('load_skill', load_skill_tool),
        ('compact', compact_tool),
        ('todo_write', todo_write_tool),
        ('read_file', read_file_tool),
        ('write_file', write_file_tool),
        ('apply_patch', apply_patch_tool),
        ('search', search_tool),
        ('run_command', run_command_tool),
        ('run_command_async', run_command_async_tool),
        ('web_search', web_search_tool),
        ('fetch_url', fetch_url_tool),
        ('analyze_memory', analyze_memory_tool),
    ]


def _builtin_git_tool_registrations() -> list[ToolRegistration]:
    """Git tool registrations."""
    from ..ops.git_tools import (
        git_branch_list_tool,
        git_checkout_tool,
        git_merge_and_push_tool,
        git_merge_tool,
        git_pull_tool,
        git_push_tool,
        git_status_tool,
    )

    return [
        ('git_status', git_status_tool),
        ('git_branch_list', git_branch_list_tool),
        ('git_checkout', git_checkout_tool),
        ('git_pull', git_pull_tool),
        ('git_merge', git_merge_tool),
        ('git_merge_and_push', git_merge_and_push_tool),
        ('git_push', git_push_tool),
    ]


def _builtin_github_tool_registrations() -> list[ToolRegistration]:
    """GitHub tool registrations."""
    from ..ops.github_tools import (
        gh_auth_status_tool,
        gh_issue_close_tool,
        gh_issue_create_tool,
        gh_issue_list_tool,
        gh_pr_checks_tool,
        gh_pr_comment_tool,
        gh_pr_create_tool,
        gh_pr_list_tool,
        gh_pr_merge_tool,
        gh_pr_view_tool,
        gh_repo_clone_tool,
        gh_repo_create_tool,
        gh_repo_list_tool,
    )

    return [
        ('gh_auth_status', gh_auth_status_tool),
        ('gh_repo_list', gh_repo_list_tool),
        ('gh_repo_create', gh_repo_create_tool),
        ('gh_repo_clone', gh_repo_clone_tool),
        ('gh_issue_list', gh_issue_list_tool),
        ('gh_issue_create', gh_issue_create_tool),
        ('gh_issue_close', gh_issue_close_tool),
        ('gh_pr_list', gh_pr_list_tool),
        ('gh_pr_create', gh_pr_create_tool),
        ('gh_pr_view', gh_pr_view_tool),
        ('gh_pr_checks', gh_pr_checks_tool),
        ('gh_pr_comment', gh_pr_comment_tool),
        ('gh_pr_merge', gh_pr_merge_tool),
    ]


def builtin_tool_registrations() -> list[ToolRegistration]:
    """All built-in tool registrations.
    
    Keep tool names stable; they are part of the model <-> harness contract.
    To add a new tool, register one more handler here. The loop itself stays unchanged.
    """
    registrations: list[ToolRegistration] = []
    registrations.extend(_builtin_core_tool_registrations())
    registrations.extend(_builtin_git_tool_registrations())
    registrations.extend(_builtin_github_tool_registrations())
    return registrations


def build_default_tools() -> ToolDispatchMap:
    """Build the default tool dispatch map."""
    return _build_tool_dispatch_map(builtin_tool_registrations())


def _risk_for_capabilities(capabilities: tuple[object, ...]) -> ToolRisk:
    """Determine risk level from capabilities."""
    if any(item.value in {'write', 'network', 'execute'} for item in capabilities):
        return ToolRisk.high
    if any(item.value == 'memory' for item in capabilities):
        return ToolRisk.medium
    return ToolRisk.low


def builtin_tool_specs() -> list[ToolSpec]:
    """Get specs for all built-in tools."""
    names = [name for name, _ in builtin_tool_registrations()]
    specs: list[ToolSpec] = []
    
    # Collect specs from each module
    all_module_specs = {
        spec.name: spec
        for spec in (
            file_tool_specs()
            + search_tool_specs()
            + command_tool_specs()
            + memory_tool_specs()
        )
    }
    
    for name in names:
        if name in all_module_specs:
            specs.append(all_module_specs[name])
        else:
            # Git/GitHub tools - generate spec from capabilities
            capabilities = TOOL_CAPABILITIES.get(name, tuple())
            specs.append(
                ToolSpec(
                    name=name,
                    description=f'{name} tool',
                    capabilities=capabilities,
                    risk_level=_risk_for_capabilities(capabilities),
                    requires_workspace=True,
                    input_notes='',
                )
            )
    return specs


def execute_tool_call(context: ToolContext, tool_call, tools: ToolDispatchMap) -> ToolResult:
    """Execute a single tool call with validation and permissions."""
    tool = tools.get(tool_call.name)
    if tool is None:
        return ToolResult(id=tool_call.id, ok=False, output='', error=f'unknown tool: {tool_call.name}')

    # Phase 1: Validate input (if tool def has validator)
    # Phase 2: Check permissions
    from ..permissions import PermissionMode
    capabilities = TOOL_CAPABILITIES.get(tool_call.name, tuple())
    if not context.policy.allows_tool(tool_call.name):
        denied = ', '.join(item.value for item in context.policy.denied_capabilities_for_tool(tool_call.name))
        return ToolResult(
            id=tool_call.id,
            ok=False,
            output='',
            error=f'tool blocked by policy: {tool_call.name} ({denied})',
            metadata={'permission_decision': 'deny', 'permission_reason': f'policy blocked capabilities: {denied}'},
        )
    permission_manager = getattr(context.policy, 'permission_manager', None)
    if permission_manager is not None:
        request = permission_manager.build_request(
            tool_name=tool_call.name,
            arguments=dict(tool_call.arguments),
            workspace_root=context.workspace_root,
            capabilities=capabilities,
        )
        decision = permission_manager.decide(request)
        if decision.mode != PermissionMode.allow:
            permission_manager.record_decision(decision.cache_key, decision.mode)
            return ToolResult(
                id=tool_call.id,
                ok=False,
                output='',
                error=f'tool blocked by permission: {tool_call.name} ({decision.reason})',
                metadata={
                    'permission_decision': decision.mode,
                    'permission_reason': decision.reason,
                    'permission_cached': decision.cached,
                },
            )
    args = dict(tool_call.arguments)
    args.setdefault('id', tool_call.id)
    result = tool(context, args)
    if permission_manager is not None:
        request = permission_manager.build_request(
            tool_name=tool_call.name,
            arguments=args,
            workspace_root=context.workspace_root,
            capabilities=capabilities,
        )
        permission_manager.record_decision(request.cache_key, PermissionMode.allow)
        result.metadata.setdefault('permission_decision', PermissionMode.allow)
        result.metadata.setdefault('permission_reason', f'{tool_call.name} allowed')
    return result
