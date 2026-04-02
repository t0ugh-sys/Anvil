"""
LoopAgent Tools - Enhanced Tool System

参考 Claude Code 工具系统设计，提供完整的工具实现。
"""
from __future__ import annotations

from typing import Any, Dict, List

from loop_agent.agent_protocol import ToolCall, ToolResult
from loop_agent.tool_def import ToolRegistration, ToolUseContext
from loop_agent.tool_builder import build_dispatch_map, execute_tool

# 导入各工具模块
from loop_agent.tools.agent import AGENT_TOOL, AgentManager, create_agent_manager, get_agent_tool
from loop_agent.tools.bash import BASH_TOOL, analyze_command, get_bash_tool
from loop_agent.tools.file_ops import get_file_tools
from loop_agent.tools.glob import GLOB_TOOL, get_glob_tool
from loop_agent.tools.grep import GREP_TOOL, get_grep_tool
from loop_agent.tools.lsp import LSP_TOOL, get_lsp_tool
from loop_agent.tools.skill import SKILL_TOOL, get_skill_tool, SkillLoader


# ============== Built-in Tool Registrations ==============

__all__ = [
    # Core types
    'ToolRegistration',
    'ToolUseContext',
    'ToolResult',
    'ToolCall',
    'AgentManager',
    'SkillLoader',
    'analyze_command',
    
    # Tool getters
    'get_grep_tool',
    'get_glob_tool',
    'get_file_tools',
    'get_bash_tool',
    'get_agent_tool',
    'get_skill_tool',
    'get_lsp_tool',
    'create_agent_manager',
    
    # Tool instances
    'GREP_TOOL',
    'GLOB_TOOL',
    'BASH_TOOL',
    'AGENT_TOOL',
    'SKILL_TOOL',
    'LSP_TOOL',
    
    # Aggregated tool list
    'builtin_tool_registrations',
    'build_tool_dispatch_map',
]


def builtin_tool_registrations() -> List[ToolRegistration]:
    """获取所有内置工具注册"""
    tools: List[ToolRegistration] = []
    
    # 文件操作工具
    tools.extend(get_file_tools())
    
    # 搜索工具
    tools.append(get_grep_tool())
    tools.append(get_glob_tool())
    
    # 命令执行
    tools.append(get_bash_tool())
    
    # Agent 管理
    tools.append(get_agent_tool())
    
    # Skill 管理
    tools.append(get_skill_tool())
    
    # LSP 支持
    tools.append(get_lsp_tool())
    
    return tools


def build_tool_dispatch_map() -> Dict[str, ToolRegistration]:
    """构建工具调度映射"""
    return build_dispatch_map(builtin_tool_registrations())


# ============== Backwards Compatibility ==============

# 保持与旧 tools.py 的兼容性
def execute_tool_call(
    context: ToolUseContext,
    tool_call: ToolCall,
    dispatch: Dict[str, ToolRegistration],
) -> ToolResult:
    """执行工具调用（兼容旧接口）- 使用 tool_builder.execute_tool）"""
    from loop_agent.tool_builder import execute_tool as exec_tool
    tool = dispatch.get(tool_call.name)
    if tool is None:
        return ToolResult(
            id=tool_call.id,
            ok=False,
            output='',
            error=f'unknown tool: {tool_call.name}',
        )
    return exec_tool(tool, context, tool_call.arguments, tool_call.id)
