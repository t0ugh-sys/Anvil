"""
GlobTool - 文件模式匹配工具

参考 Claude Code GlobTool 的实现。
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from loop_agent.agent_protocol import ToolResult
from loop_agent.tool_def import ToolRegistration, ToolUseContext
from loop_agent.tool_builder import build_tool, build_search_tool


# ============== Skip Patterns ==============

GLOB_SKIP_DIRS = {
    '.git',
    '.loopagent',
    '.mypy_cache',
    '.pytest_cache',
    '.ruff_cache',
    '.venv',
    '__pycache__',
    'build',
    'dist',
    'node_modules',
    'vendor',
}


# ============== Glob Implementation ==============

def _glob_paths(
    workspace_root: Path,
    pattern: str,
    limit: int = 100,
) -> List[str]:
    """使用 fnmatch 进行文件匹配"""
    import fnmatch
    
    paths: List[str] = []
    
    for root, dirs, filenames in os.walk(workspace_root):
        # 跳过目录
        dirs[:] = [d for d in dirs if d not in GLOB_SKIP_DIRS]
        
        root_path = Path(root)
        
        for filename in filenames:
            if fnmatch.fnmatch(filename, pattern):
                full_path = root_path / filename
                relative = full_path.relative_to(workspace_root)
                paths.append(relative.as_posix())
                
                if len(paths) >= limit:
                    return paths
    
    return paths


def _format_glob_output(
    paths: List[str],
    limit: int = 100,
) -> str:
    """格式化 glob 输出"""
    if not paths:
        return 'No files found'
    
    truncated = len(paths) > limit
    display_paths = paths[:limit]
    
    lines = [f'Found {len(display_paths)} files:']
    lines.extend(f'  {p}' for p in display_paths)
    
    if truncated:
        lines.append(f'... (truncated, {len(paths)} total)')
    
    return '\n'.join(lines)


# ============== GlobTool Handler ==============

def _glob_handler(
    context: ToolUseContext,
    args: Dict[str, Any],
) -> ToolResult:
    """Glob 工具处理函数"""
    call_id = str(args.get('id', 'glob'))
    pattern = str(args.get('pattern', '')).strip()
    
    if not pattern:
        return ToolResult(id=call_id, ok=False, output='', error='pattern is required')
    
    workspace = Path(context.workspace_root)
    
    # 可选参数
    path_filter = str(args.get('path', '')).strip() or None
    limit = int(args.get('limit', 100))
    
    # 实际搜索目录
    if path_filter:
        search_root = workspace / path_filter
        if not search_root.exists():
            return ToolResult(id=call_id, ok=False, output='', error=f'path not found: {path_filter}')
    else:
        search_root = workspace
    
    try:
        paths = _glob_paths(search_root, pattern, limit)
        output = _format_glob_output(paths, limit)
        return ToolResult(id=call_id, ok=True, output=output, error=None)
        
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


# ============== Tool Registration ==============

GLOB_TOOL = build_search_tool(
    name='glob',
    description='Find files matching a pattern using glob',
    handler=_glob_handler,
    input_schema={
        'type': 'object',
        'properties': {
            'pattern': {
                'type': 'string',
                'description': 'Glob pattern (e.g., "*.py", "src/**/*.ts")',
            },
            'path': {
                'type': 'string',
                'description': 'Directory to search in (default: workspace root)',
            },
            'limit': {
                'type': 'number',
                'description': 'Maximum number of results',
                'default': 100,
            },
        },
        'required': ['pattern'],
    },
    search_hint='find files by pattern',
)


# ============== Exports ==============

def get_glob_tool() -> ToolRegistration:
    return GLOB_TOOL
