"""
Tool Builder - Factory for Creating Tools

参考 Claude Code 的 buildTool 工厂函数，统一创建 Tool 实例。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

from loop_agent.agent_protocol import ToolCall, ToolResult
from loop_agent.tool_def import (
    ToolDefaults,
    ToolRegistration,
    ToolUseContext,
    ValidationResult,
)


# ============== Build Options ==============

ToolDef = Dict[str, Any]


# ============== Default Implementations ==============

def _default_is_enabled() -> bool:
    return True


def _default_is_concurrency_safe(_input: Any) -> bool:
    return False


def _default_is_read_only(_input: Any) -> bool:
    return False


def _default_is_destructive(_input: Any) -> bool:
    return False


def _default_check_permissions(_input: Any, _context: ToolUseContext) -> bool:
    return True


def _default_user_facing_name(_input: Optional[Dict]) -> str:
    return ''


def _default_to_auto_classifier_input(_input: Any) -> str:
    return ''


# ============== build_tool Factory ==============

def build_tool(
    defn: ToolDef,
) -> ToolRegistration:
    """
    构建完整的 Tool 实例。
    
    参考 Claude Code 的 buildTool 设计，提供默认实现填充。
    
    Args:
        defn: Tool 定义字典，包含:
            - name: str (必需) - Tool 名称
            - handler: callable (必需) - 执行函数
            - input_schema: dict - 输入 schema
            - output_schema: dict - 输出 schema
            - description: str - 描述
            - search_hint: str - 搜索提示
            - is_read_only: bool - 是否只读
            - is_destructive: bool - 是否具破坏性
            - max_result_size_chars: int - 最大结果大小
            - should_defer: bool - 是否延迟加载
            - always_load: bool - 是否始终加载
    
    Returns:
        ToolRegistration - 完整的 Tool 实例
    """
    # 必需字段
    name = defn.get('name')
    if not name:
        raise ValueError('Tool definition must have a "name" field')
    
    handler = defn.get('handler')
    if not handler:
        raise ValueError(f'Tool "{name}" must have a "handler" function')
    
    # 填充默认值
    registration = ToolRegistration(
        name=name,
        handler=handler,
        input_schema=defn.get('input_schema', {}),
        output_schema=defn.get('output_schema'),
        description=defn.get('description', ''),
        search_hint=defn.get('search_hint', ''),
        max_result_size_chars=defn.get('max_result_size_chars', 100_000),
        should_defer=defn.get('should_defer', False),
        always_load=defn.get('always_load', False),
        is_read_only=defn.get('is_read_only', False),
        is_destructive=defn.get('is_destructive', False),
    )
    
    return registration


# ============== Convenience Builders ==============

def build_read_tool(
    name: str,
    description: str,
    handler: Callable[[ToolUseContext, Dict[str, Any]], ToolResult],
    **kwargs,
) -> ToolRegistration:
    """构建只读 Tool"""
    return build_tool({
        'name': name,
        'description': description,
        'handler': handler,
        'is_read_only': True,
        **kwargs,
    })


def build_write_tool(
    name: str,
    description: str,
    handler: Callable[[ToolUseContext, Dict[str, Any]], ToolResult],
    **kwargs,
) -> ToolRegistration:
    """构建写入 Tool"""
    return build_tool({
        'name': name,
        'description': description,
        'handler': handler,
        'is_read_only': False,
        'is_destructive': kwargs.get('is_destructive', False),
        **kwargs,
    })


def build_search_tool(
    name: str,
    description: str,
    handler: Callable[[ToolUseContext, Dict[str, Any]], ToolResult],
    **kwargs,
) -> ToolRegistration:
    """构建搜索 Tool"""
    return build_tool({
        'name': name,
        'description': description,
        'handler': handler,
        'search_hint': kwargs.get('search_hint', f'{name} search'),
        'is_read_only': True,
        **kwargs,
    })


# ============== Schema Helpers ==============

def string_schema(**kwargs) -> Dict[str, Any]:
    """构建字符串类型 schema"""
    return {
        'type': 'object',
        'properties': {
            'path': {'type': 'string', 'description': kwargs.get('path_desc', 'File path')},
        },
        'required': kwargs.get('required', ['path']),
    }


def optional_string_schema(**kwargs) -> Dict[str, Any]:
    """构建可选字符串类型 schema"""
    props = {
        'path': {'type': 'string', 'description': kwargs.get('path_desc', 'File path')},
    }
    for k, v in kwargs.get('extra', {}).items():
        props[k] = v
    
    return {
        'type': 'object',
        'properties': props,
    }


# ============== Tool Execution ==============

def execute_tool(
    tool: ToolRegistration,
    context: ToolUseContext,
    args: Dict[str, Any],
    call_id: Optional[str] = None,
) -> ToolResult:
    """
    执行 Tool 并返回结果。
    
    包含完整的错误处理和验证流程。
    """
    # 检查是否启用
    # if hasattr(tool, 'is_enabled') and not tool.is_enabled():
    #     return ToolResult(
    #         id=call_id or 'unknown',
    #         ok=False,
    #         output='',
    #         error=f'Tool "{tool.name}" is disabled',
    #     )
    
    try:
        # 添加 call_id 到参数
        args = dict(args)
        args.setdefault('id', call_id)
        
        # 执行 handler
        result = tool.handler(context, args)
        
        # 确保返回 ToolResult
        if not isinstance(result, ToolResult):
            return ToolResult(
                id=call_id or 'unknown',
                ok=False,
                output='',
                error=f'Handler returned invalid type: {type(result)}',
            )
        
        return result
        
    except Exception as exc:
        return ToolResult(
            id=call_id or 'unknown',
            ok=False,
            output='',
            error=str(exc),
        )


def validate_tool_input(
    tool: ToolRegistration,
    args: Dict[str, Any],
) -> ValidationResult:
    """
    验证 Tool 输入参数。
    
    参考 Claude Code 的 validateInput 机制。
    """
    schema = tool.input_schema
    
    # 简单验证 - 检查必需字段
    required = schema.get('required', [])
    for field in required:
        if field not in args:
            return ValidationResult(
                result=False,
                message=f'Missing required field: {field}',
                error_code=400,
            )
    
    return ValidationResult(result=True)


# ============== Dispatch Map ==============

def build_dispatch_map(
    tools: List[ToolRegistration],
) -> Dict[str, ToolRegistration]:
    """从 Tool 列表构建调度映射"""
    return {tool.name: tool for tool in tools}


def execute_tool_dispatch(
    dispatch: Dict[str, ToolRegistration],
    context: ToolUseContext,
    tool_call: ToolCall,
) -> ToolResult:
    """
    通过名称调度执行 Tool。
    
    类似 Claude Code 的工具路由机制。
    """
    tool = dispatch.get(tool_call.name)
    
    if tool is None:
        return ToolResult(
            id=tool_call.id,
            ok=False,
            output='',
            error=f'unknown tool: {tool_call.name}',
        )
    
    return execute_tool(tool, context, tool_call.arguments, tool_call.id)
