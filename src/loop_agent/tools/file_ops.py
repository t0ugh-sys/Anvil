"""
FileEditTool - 文件编辑工具

参考 Claude Code FileEditTool 的实现，支持原地编辑和 diff。
"""
from __future__ import annotations

import difflib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loop_agent.agent_protocol import ToolResult
from loop_agent.tool_def import ToolRegistration, ToolUseContext
from loop_agent.tool_builder import build_tool, build_write_tool


# ============== Edit Operations ==============

def _resolve_inside_workspace(workspace_root: Path, relative_path: str) -> Path:
    """解析工作区内的路径"""
    root = workspace_root.resolve()
    target = (workspace_root / relative_path).resolve()
    if os.path.commonpath([str(root), str(target)]) != str(root):
        raise ValueError('path escapes workspace root')
    return target


def _find_edit_position(
    content: str,
    search_string: str,
) -> Optional[Tuple[int, int]]:
    """
    查找编辑位置。
    
    Returns:
        (start_line, end_line) 或 None
    """
    lines = content.split('\n')
    search_lines = search_string.split('\n')
    
    # 简单字符串匹配
    search_text = search_string.strip()
    
    for i in range(len(lines)):
        # 检查从这一行开始的匹配
        remaining = '\n'.join(lines[i:])
        if search_text in remaining:
            # 找到起始位置
            start = i
            # 找到结束位置
            for j, search_line in enumerate(search_lines):
                if i + j >= len(lines):
                    break
                if lines[i + j] != search_line:
                    break
            else:
                # 完整匹配
                end = i + len(search_lines)
                return (start, end)
    
    return None


def _apply_string_edit(
    content: str,
    search_string: str,
    replacement_string: str,
) -> str:
    """应用字符串编辑"""
    lines = content.split('\n')
    search_lines = search_string.split('\n')
    replace_lines = replacement_string.split('\n')
    
    # 查找匹配
    for i in range(len(lines)):
        if i + len(search_lines) > len(lines):
            continue
        
        # 检查是否匹配
        match = True
        for j, search_line in enumerate(search_lines):
            if lines[i + j].rstrip() != search_line.rstrip():
                match = False
                break
        
        if match:
            # 应用编辑
            new_lines = lines[:i] + replace_lines + lines[i + len(search_lines):]
            return '\n'.join(new_lines)
    
    # 没找到匹配，尝试更宽松的匹配
    search_text = search_string.strip()
    content_stripped = content.strip()
    
    if search_text in content_stripped:
        # 使用 difflib
        content_lines = content.split('\n')
        search_lines_stripped = [l.rstrip() for l in search_lines if l.strip()]
        
        for i in range(len(content_lines)):
            if any(search_lines_stripped[0] in l for l in content_lines[i:]):
                # 找到大致位置，尝试智能替换
                pass
    
    raise ValueError('search string not found in file')


def _generate_diff(
    original: str,
    replacement: str,
    path: str,
) -> str:
    """生成 unified diff"""
    original_lines = original.split('\n')
    replacement_lines = replacement.split('\n')
    
    diff = difflib.unified_diff(
        original_lines,
        replacement_lines,
        fromfile=path,
        tofile=path,
        lineterm='',
    )
    
    return '\n'.join(diff)


# ============== EditTool Handler ==============

def _edit_handler(
    context: ToolUseContext,
    args: Dict[str, Any],
) -> ToolResult:
    """FileEdit 工具处理函数"""
    call_id = str(args.get('id', 'edit'))
    
    path = str(args.get('path', '')).strip()
    if not path:
        return ToolResult(id=call_id, ok=False, output='', error='path is required')
    
    search_string = str(args.get('search_string', ''))
    replacement_string = str(args.get('replacement_string', ''))
    
    if not search_string and not replacement_string:
        return ToolResult(id=call_id, ok=False, output='', error='search_string is required')
    
    workspace = Path(context.workspace_root)
    
    try:
        target = _resolve_inside_workspace(workspace, path)
        
        if not target.exists():
            return ToolResult(id=call_id, ok=False, output='', error=f'file not found: {path}')
        
        # 读取原内容
        original = target.read_text(encoding='utf-8')
        
        # 应用编辑
        if search_string and replacement_string:
            updated = _apply_string_edit(original, search_string, replacement_string)
        elif replacement_string:
            # 整个文件替换
            updated = replacement_string
        else:
            # 只删除
            return ToolResult(id=call_id, ok=False, output='', error='replacement_string is required')
        
        # 写回文件
        target.write_text(updated, encoding='utf-8')
        
        # 生成 diff（可选）
        show_diff = args.get('show_diff', False)
        if show_diff:
            diff = _generate_diff(original, updated, path)
            return ToolResult(id=call_id, ok=True, output=diff, error=None)
        
        return ToolResult(id=call_id, ok=True, output=f'Edited {path}', error=None)
        
    except ValueError as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def _read_handler(
    context: ToolUseContext,
    args: Dict[str, Any],
) -> ToolResult:
    """FileRead 工具处理函数"""
    call_id = str(args.get('id', 'read'))
    
    path = str(args.get('path', '')).strip()
    if not path:
        return ToolResult(id=call_id, ok=False, output='', error='path is required')
    
    workspace = Path(context.workspace_root)
    
    try:
        target = _resolve_inside_workspace(workspace, path)
        
        if not target.exists():
            return ToolResult(id=call_id, ok=False, output='', error=f'file not found: {path}')
        
        # 读取内容
        content = target.read_text(encoding='utf-8')
        
        # 可选的偏移和限制
        offset = int(args.get('offset', 0))
        limit = int(args.get('limit', 0))
        
        lines = content.split('\n')
        
        if offset > 0:
            lines = lines[offset:]
        if limit > 0:
            lines = lines[:limit]
        
        return ToolResult(id=call_id, ok=True, output='\n'.join(lines), error=None)
        
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def _write_handler(
    context: ToolUseContext,
    args: Dict[str, Any],
) -> ToolResult:
    """FileWrite 工具处理函数"""
    call_id = str(args.get('id', 'write'))
    
    path = str(args.get('path', '')).strip()
    content = str(args.get('content', ''))
    
    if not path:
        return ToolResult(id=call_id, ok=False, output='', error='path is required')
    
    workspace = Path(context.workspace_root)
    
    try:
        target = _resolve_inside_workspace(workspace, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
        
        return ToolResult(id=call_id, ok=True, output=f'Wrote {path}', error=None)
        
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


# ============== Tool Registrations ==============

EDIT_TOOL = build_write_tool(
    name='edit',
    description='Edit a file by replacing specific text',
    handler=_edit_handler,
    is_destructive=True,
    input_schema={
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'File path to edit',
            },
            'search_string': {
                'type': 'string',
                'description': 'Text to find and replace',
            },
            'replacement_string': {
                'type': 'string',
                'description': 'Replacement text',
            },
            'show_diff': {
                'type': 'boolean',
                'description': 'Show unified diff of changes',
                'default': False,
            },
        },
        'required': ['path'],
    },
)

READ_TOOL = build_tool({
    'name': 'read',
    'description': 'Read file contents',
    'handler': _read_handler,
    'is_read_only': True,
    'input_schema': {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'File path to read',
            },
            'offset': {
                'type': 'number',
                'description': 'Line offset to start reading from',
                'default': 0,
            },
            'limit': {
                'type': 'number',
                'description': 'Maximum number of lines to read',
                'default': 0,
            },
        },
        'required': ['path'],
    },
})

WRITE_TOOL = build_write_tool(
    name='write',
    description='Write content to a file',
    handler=_write_handler,
    is_destructive=True,
    input_schema={
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'File path to write',
            },
            'content': {
                'type': 'string',
                'description': 'Content to write',
            },
        },
        'required': ['path', 'content'],
    },
)


# ============== Exports ==============

def get_file_tools() -> List[ToolRegistration]:
    return [EDIT_TOOL, READ_TOOL, WRITE_TOOL]
