"""
GrepTool - 正则表达式搜索工具

参考 Claude Code GrepTool 的实现。
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from loop_agent.agent_protocol import ToolResult
from loop_agent.tool_def import ToolRegistration, ToolUseContext
from loop_agent.tool_builder import build_tool, build_search_tool


# ============== Skip Patterns ==============

GREP_SKIP_DIRS = {
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


# ============== Grep Result ==============

@dataclass
class GrepMatch:
    """Grep 匹配结果"""
    path: str
    line_number: int
    line: str
    match_start: int
    match_end: int


def _iter_files(
    workspace_root: Path,
    pattern: str,
    include: Optional[str] = None,
    exclude: Optional[str] = None,
) -> List[Path]:
    """迭代匹配的文件"""
    files: List[Path] = []
    
    include_re = re.compile(include) if include else None
    exclude_re = re.compile(exclude) if exclude else None
    
    for root, dirs, filenames in os.walk(workspace_root):
        # 跳过目录
        dirs[:] = [d for d in dirs if d not in GREP_SKIP_DIRS]
        
        root_path = Path(root)
        for filename in filenames:
            file_path = root_path / filename
            
            # 跳过二进制文件（简单检查）
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.read(1024)
            except (UnicodeDecodeError, ValueError):
                continue
            
            # 过滤 include/exclude
            if include_re and not include_re.search(filename):
                continue
            if exclude_re and exclude_re.search(filename):
                continue
            
            files.append(file_path)
    
    return files


def _search_file(
    path: Path,
    pattern: str,
    is_regex: bool = True,
) -> List[GrepMatch]:
    """在单个文件中搜索"""
    matches: List[GrepMatch] = []
    
    try:
        content = path.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return matches
    
    # 编译正则
    try:
        if is_regex:
            regex = re.compile(pattern)
        else:
            regex = re.compile(re.escape(pattern))
    except re.error:
        return matches
    
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        for match in regex.finditer(line):
            matches.append(GrepMatch(
                path=str(path),
                line_number=i,
                line=line,
                match_start=match.start(),
                match_end=match.end(),
            ))
    
    return matches


def _format_grep_output(
    matches: List[GrepMatch],
    max_results: int = 100,
) -> str:
    """格式化 grep 输出"""
    if not matches:
        return 'No matches found'
    
    # 限制结果数量
    display_matches = matches[:max_results]
    truncated = len(matches) > max_results
    
    lines: List[str] = []
    current_file: Optional[str] = None
    
    for match in display_matches:
        if match.path != current_file:
            current_file = match.path
            lines.append(f'\n{match.path}:')
        
        # 显示行号和内容
        line = match.line.rstrip()
        lines.append(f'  {match.line_number}: {line}')
    
    if truncated:
        lines.append(f'\n... (truncated, {len(matches)} total matches)')
    
    return '\n'.join(lines)


# ============== GrepTool Handler ==============

def _grep_handler(
    context: ToolUseContext,
    args: Dict[str, Any],
) -> ToolResult:
    """Grep 工具处理函数"""
    call_id = str(args.get('id', 'grep'))
    pattern = str(args.get('pattern', '')).strip()
    
    if not pattern:
        return ToolResult(id=call_id, ok=False, output='', error='pattern is required')
    
    workspace = Path(context.workspace_root)
    
    # 可选参数
    is_regex = args.get('is_regex', True)
    include = args.get('include', '').strip() or None
    exclude = args.get('exclude', '').strip() or None
    context_lines = int(args.get('context_lines', 0))
    max_results = int(args.get('max_results', 100))
    
    try:
        # 先尝试使用系统 grep（更高效）
        cmd = ['grep', '-rn', '--include=*', pattern, '.']
        
        result = subprocess.run(
            cmd,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            shell=False,
        )
        
        # 如果 grep 成功使用它
        if result.returncode in (0, 1):  # 0=找到, 1=未找到
            output = result.stdout.strip()
            if not output:
                return ToolResult(id=call_id, ok=True, output='No matches found', error=None)
            
            # 格式化输出
            lines = output.split('\n')[:max_results]
            truncated = len(output.split('\n')) > max_results
            
            formatted = '\n'.join(lines)
            if truncated:
                formatted += f'\n... (truncated)'
            
            return ToolResult(id=call_id, ok=True, output=formatted, error=None)
        
        # 回退到纯 Python 实现
        matches: List[GrepMatch] = []
        
        for file_path in _iter_files(workspace, pattern, include, exclude):
            file_matches = _search_file(file_path, pattern, is_regex)
            matches.extend(file_matches)
            
            if len(matches) >= max_results:
                break
        
        output = _format_grep_output(matches, max_results)
        return ToolResult(id=call_id, ok=True, output=output, error=None)
        
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


# ============== Tool Registration ==============

GREP_TOOL = build_search_tool(
    name='grep',
    description='Search for patterns in files using regular expressions',
    handler=_grep_handler,
    input_schema={
        'type': 'object',
        'properties': {
            'pattern': {
                'type': 'string',
                'description': 'Regular expression pattern to search for',
            },
            'path': {
                'type': 'string',
                'description': 'Directory to search in (default: workspace root)',
            },
            'is_regex': {
                'type': 'boolean',
                'description': 'Treat pattern as regex (default: true)',
                'default': True,
            },
            'include': {
                'type': 'string',
                'description': 'File pattern to include (e.g., "*.py")',
            },
            'exclude': {
                'type': 'string',
                'description': 'File pattern to exclude',
            },
            'context_lines': {
                'type': 'number',
                'description': 'Number of context lines to show',
                'default': 0,
            },
            'max_results': {
                'type': 'number',
                'description': 'Maximum number of results',
                'default': 100,
            },
        },
        'required': ['pattern'],
    },
    search_hint='search code patterns',
)


# ============== Exports ==============

def get_grep_tool() -> ToolRegistration:
    return GREP_TOOL
