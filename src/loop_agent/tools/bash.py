"""
BashTool - Shell Command Execution with Security

参考 Claude Code BashTool 的设计，包含权限检查和安全验证。
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from loop_agent.agent_protocol import ToolResult
from loop_agent.tool_def import ToolRegistration, ToolUseContext
from loop_agent.tool_builder import build_tool, build_write_tool


# ============== Security Rules ==============

# 危险命令模式
DANGEROUS_PATTERNS = [
    r'^\s*rm\s+-rf\s+',
    r'^\s*rm\s+/\s',
    r'^\s*mkfs\.',
    r'^\s*dd\s+if=',
    r'>\s*/dev/sd',
    r'>\s*/dev/null\s*>\s*/dev/null',
    r'curl.*\|\s*bash',
    r'wget.*\|\s*bash',
    r':\(\)\{',  # Fork bomb
    r'chmod\s+-R\s+777',
    r'chown\s+-R',
]

# 只读命令模式（无需确认）
READONLY_COMMANDS = {
    'ls', 'la', 'll', 'pwd', 'cat', 'head', 'tail', 'less', 'more',
    'find', 'grep', 'rg', 'ag', 'which', 'whereis', 'file', 'stat',
    'tree', 'du', 'df', 'wc', 'sort', 'uniq', 'cut', 'awk', 'sed',
    'git', 'git status', 'git log', 'git show', 'git diff',
    'npm', 'pip', 'python', 'node', 'ruby', 'perl',
}

# 需要确认的命令
DESTRUCTIVE_KEYWORDS = {
    'rm', 'del', 'rmdir', 'format', 'mkfs',
    'mv', 'cp', 'install',
    'chmod', 'chown', 'chgrp',
    'dd', 'fdisk', 'parted',
    'shutdown', 'reboot', 'halt',
    'kill', 'killall', 'pkill',
    'echo', 'printf', 'tee',
}


# ============== Command Classifier ==============

@dataclass
class CommandAnalysis:
    """命令分析结果"""
    is_readonly: bool = False
    is_dangerous: bool = False
    is_shell_injection_risk: bool = False
    needs_confirmation: bool = False
    warnings: List[str] = field(default_factory=list)


def _analyze_command(command: str) -> CommandAnalysis:
    """分析命令的安全性"""
    analysis = CommandAnalysis()
    
    # 清理命令（移除管道等）
    clean_cmd = command.split('|')[0].split('&&')[0].split(';')[0].strip()
    base_cmd = clean_cmd.split()[0] if clean_cmd else ''
    
    # 检查只读
    if base_cmd in READONLY_COMMANDS:
        analysis.is_readonly = True
    
    # 检查危险模式
    for pattern in DANGEROUS_PATTERNS:
        if re.match(pattern, command, re.IGNORECASE):
            analysis.is_dangerous = True
            analysis.warnings.append(f'Match dangerous pattern: {pattern}')
    
    # 检查需要确认的命令
    for keyword in DESTRUCTIVE_KEYWORDS:
        if keyword in clean_cmd.lower():
            analysis.needs_confirmation = True
    
    # 检查 shell 注入风险
    if '$(' in command or '`' in command or '${' in command:
        analysis.is_shell_injection_risk = True
        analysis.warnings.append('Command contains shell injection risk')
    
    return analysis


def _validate_path_arg(path_arg: str, workspace_root: Path) -> Optional[str]:
    """验证路径参数安全性"""
    if not path_arg:
        return None
    
    # 尝试解析为路径
    try:
        target = (workspace_root / path_arg).resolve()
        
        # 检查是否在工作区内
        root = workspace_root.resolve()
        if os.path.commonpath([str(root), str(target)]) != str(root):
            return f'Path escapes workspace: {path_arg}'
        
        return None
    except Exception as e:
        return f'Invalid path: {path_arg} ({e})'


# ============== BashTool Handler ==============

def _bash_handler(
    context: ToolUseContext,
    args: Dict[str, Any],
) -> ToolResult:
    """Bash 工具处理函数"""
    call_id = str(args.get('id', 'bash'))
    
    command = str(args.get('command', '')).strip()
    if not command:
        return ToolResult(id=call_id, ok=False, output='', error='command is required')
    
    workspace = Path(context.workspace_root)
    
    # 安全检查
    analysis = _analyze_command(command)
    
    # 阻止危险命令
    if analysis.is_dangerous:
        return ToolResult(
            id=call_id,
            ok=False,
            output='',
            error=f'Dangerous command blocked: {analysis.warnings[0]}',
        )
    
    # 权限检查
    policy = context.permission_context
    if policy.mode == 'readonly':
        if not analysis.is_readonly:
            return ToolResult(
                id=call_id,
                ok=False,
                output='',
                error='Command blocked: readonly mode',
            )
    
    # 检查规则匹配
    tool_name = 'bash'
    if tool_name in policy.always_allow_rules:
        allowed = policy.always_allow_rules[tool_name]
        # 检查是否有匹配的模式
        for pattern in allowed:
            if _matches_pattern(command, pattern):
                analysis.is_readonly = True
                break
    
    if tool_name in policy.always_deny_rules:
        denied = policy.always_deny_rules[tool_name]
        for pattern in denied:
            if _matches_pattern(command, pattern):
                return ToolResult(
                    id=call_id,
                    ok=False,
                    output='',
                    error=f'Command blocked by policy: {pattern}',
                )
    
    try:
        # 确定是否使用 shell
        use_shell = _needs_shell(command)
        
        proc = subprocess.run(
            command,
            cwd=str(workspace),
            shell=use_shell,
            check=False,
            text=True,
            capture_output=True,
            encoding='utf-8',
            errors='replace',
            timeout=int(args.get('timeout', 300)),
        )
        
        merged = (proc.stdout or '') + (proc.stderr or '')
        ok = proc.returncode == 0
        
        if ok:
            return ToolResult(id=call_id, ok=True, output=merged.strip(), error=None)
        else:
            return ToolResult(
                id=call_id,
                ok=False,
                output=merged.strip(),
                error=f'exit code: {proc.returncode}',
            )
            
    except subprocess.TimeoutExpired:
        return ToolResult(id=call_id, ok=False, output='', error='command timeout')
    except FileNotFoundError as exc:
        return ToolResult(id=call_id, ok=False, output='', error=f'command not found: {exc.filename}')
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def _matches_pattern(command: str, pattern: str) -> bool:
    """检查命令是否匹配模式"""
    # 简单实现：支持通配符
    import fnmatch
    return fnmatch.fnmatch(command, pattern)


def _needs_shell(command: str) -> bool:
    """检查命令是否需要 shell 才能执行"""
    # 管道、重定向、后台等需要 shell
    special_chars = ['|', '&', ';', '>', '<', '(', ')']
    return any(char in command for char in special_chars)


# ============== Tool Registration ==============

BASH_TOOL = build_write_tool(
    name='bash',
    description='Execute shell commands',
    handler=_bash_handler,
    is_destructive=True,
    input_schema={
        'type': 'object',
        'properties': {
            'command': {
                'type': 'string',
                'description': 'Shell command to execute',
            },
            'timeout': {
                'type': 'number',
                'description': 'Timeout in seconds',
                'default': 300,
            },
        },
        'required': ['command'],
    },
    search_hint='run terminal commands',
)


# ============== Exports ==============

def get_bash_tool() -> ToolRegistration:
    return BASH_TOOL


def analyze_command(command: str) -> CommandAnalysis:
    """公开的命令分析函数"""
    return _analyze_command(command)
