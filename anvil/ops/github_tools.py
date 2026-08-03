"""Backward-compatibility shim — re-exports everything from the github subpackage."""
from __future__ import annotations

from .github import (
    GhOptions,
    _merge_output,
    _normalize_repo,
    _parse_repo_from_remote,
    _require_gh_available,
    _resolve_repo_arg,
    _run,
    _run_gh,
    _strip_suffix,
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

__all__ = [
    'gh_auth_status_tool',
    'gh_repo_list_tool',
    'gh_repo_create_tool',
    'gh_repo_clone_tool',
    'gh_issue_list_tool',
    'gh_issue_create_tool',
    'gh_issue_close_tool',
    'gh_pr_list_tool',
    'gh_pr_create_tool',
    'gh_pr_view_tool',
    'gh_pr_checks_tool',
    'gh_pr_comment_tool',
    'gh_pr_merge_tool',
]
