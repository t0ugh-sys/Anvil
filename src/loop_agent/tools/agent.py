"""
AgentTool - Sub-Agent Management System

参考 Claude Code AgentTool 的设计，支持子 Agent 定义、调度和管理。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loop_agent.agent_protocol import ToolCall, ToolResult
from loop_agent.tool_def import ToolRegistration, ToolUseContext
from loop_agent.tool_builder import build_tool


# ============== Agent Definition ==============

@dataclass
class AgentDefinition:
    """子 Agent 定义"""
    id: str
    name: str
    description: str
    system_prompt: str
    model: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    color: Optional[str] = None
    is_builtin: bool = False
    is_custom: bool = False


@dataclass
class AgentInstance:
    """运行的 Agent 实例"""
    id: str
    definition: AgentDefinition
    state: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0


# ============== Agent Loader ==============

def _load_builtin_agents() -> Dict[str, AgentDefinition]:
    """加载内置 Agent"""
    agents: Dict[str, AgentDefinition] = {}
    
    # Coding Agent - 默认的编码 Agent
    agents['default'] = AgentDefinition(
        id='default',
        name='default',
        description='Default coding agent',
        system_prompt='You are a coding assistant. Write, edit, and review code as needed.',
        is_builtin=True,
    )
    
    # Review Agent - 代码审查
    agents['review'] = AgentDefinition(
        id='review',
        name='review',
        description='Code review agent',
        system_prompt='You are a code review assistant. Review code for bugs, performance, and style issues.',
        is_builtin=True,
    )
    
    # Debug Agent - 调试 Agent
    agents['debug'] = AgentDefinition(
        id='debug',
        name='debug',
        description='Debugging assistant',
        system_prompt='You are a debugging assistant. Find and fix bugs in code.',
        is_builtin=True,
    )
    
    return agents


def _load_agents_from_dir(agents_dir: Path) -> Dict[str, AgentDefinition]:
    """从目录加载自定义 Agent"""
    agents: Dict[str, AgentDefinition] = {}
    
    if not agents_dir.exists():
        return agents
    
    # 加载 JSON 格式的 Agent 定义
    for json_file in agents_dir.glob('*.json'):
        try:
            import json
            data = json.loads(json_file.read_text(encoding='utf-8'))
            
            agent = AgentDefinition(
                id=data.get('id', json_file.stem),
                name=data.get('name', json_file.stem),
                description=data.get('description', ''),
                system_prompt=data.get('system_prompt', ''),
                model=data.get('model'),
                tools=data.get('tools', []),
                color=data.get('color'),
                is_custom=True,
            )
            
            agents[agent.id] = agent
            
        except Exception:
            continue
    
    return agents


# ============== Agent Manager ==============

class AgentManager:
    """Agent 管理器"""
    
    def __init__(self, agents_dir: Optional[Path] = None):
        self._builtin_agents = _load_builtin_agents()
        self._custom_agents = _load_agents_from_dir(agents_dir) if agents_dir else {}
        self._running_instances: Dict[str, AgentInstance] = {}
    
    def get_all_agents(self) -> List[AgentDefinition]:
        """获取所有可用的 Agent"""
        agents = list(self._builtin_agents.values())
        agents.extend(list(self._custom_agents.values()))
        return agents
    
    def get_agent(self, agent_id: str) -> Optional[AgentDefinition]:
        """获取 Agent 定义"""
        return self._builtin_agents.get(agent_id) or self._custom_agents.get(agent_id)
    
    def register_agent(self, definition: AgentDefinition) -> None:
        """注册自定义 Agent"""
        self._custom_agents[definition.id] = definition
    
    def create_instance(self, agent_id: str) -> Optional[AgentInstance]:
        """创建 Agent 实例"""
        definition = self.get_agent(agent_id)
        if not definition:
            return None
        
        instance = AgentInstance(
            id=str(uuid.uuid4()),
            definition=definition,
            created_at=0.0,  # TODO: 添加时间戳
        )
        
        self._running_instances[instance.id] = instance
        return instance
    
    def get_instance(self, instance_id: str) -> Optional[AgentInstance]:
        """获取 Agent 实例"""
        return self._running_instances.get(instance_id)
    
    def terminate_instance(self, instance_id: str) -> bool:
        """终止 Agent 实例"""
        if instance_id in self._running_instances:
            del self._running_instances[instance_id]
            return True
        return False


# ============== AgentTool Handler ==============

def _agent_handler(
    context: ToolUseContext,
    args: Dict[str, Any],
) -> ToolResult:
    """Agent 工具处理函数"""
    call_id = str(args.get('id', 'agent'))
    
    action = str(args.get('action', 'list')).strip().lower()
    
    # 获取 Agent 管理器
    agent_manager = getattr(context, '_agent_manager', None)
    if agent_manager is None:
        return ToolResult(
            id=call_id,
            ok=False,
            output='',
            error='Agent manager not configured',
        )
    
    try:
        if action == 'list':
            # 列出所有 Agent
            agents = agent_manager.get_all_agents()
            lines = ['Available agents:']
            for agent in agents:
                lines.append(f'  - {agent.id}: {agent.description}')
            
            return ToolResult(id=call_id, ok=True, output='\n'.join(lines), error=None)
        
        elif action == 'create':
            # 创建 Agent 实例
            agent_id = str(args.get('agent_id', '')).strip()
            if not agent_id:
                return ToolResult(id=call_id, ok=False, output='', error='agent_id is required')
            
            instance = agent_manager.create_instance(agent_id)
            if not instance:
                return ToolResult(id=call_id, ok=False, output='', error=f'Unknown agent: {agent_id}')
            
            return ToolResult(
                id=call_id,
                ok=True,
                output=f'Created agent instance: {instance.id}',
                error=None,
            )
        
        elif action == 'terminate':
            # 终止 Agent 实例
            instance_id = str(args.get('instance_id', '')).strip()
            if not instance_id:
                return ToolResult(id=call_id, ok=False, output='', error='instance_id is required')
            
            success = agent_manager.terminate_instance(instance_id)
            if success:
                return ToolResult(id=call_id, ok=True, output=f'Terminated: {instance_id}', error=None)
            else:
                return ToolResult(id=call_id, ok=False, output='', error=f'Instance not found: {instance_id}')
        
        else:
            return ToolResult(id=call_id, ok=False, output='', error=f'Unknown action: {action}')
    
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


# ============== Tool Registration ==============

AGENT_TOOL = build_tool({
    'name': 'agent',
    'description': 'Manage and invoke sub-agents',
    'handler': _agent_handler,
    'input_schema': {
        'type': 'object',
        'properties': {
            'action': {
                'type': 'string',
                'description': 'Action: list, create, terminate',
                'enum': ['list', 'create', 'terminate'],
            },
            'agent_id': {
                'type': 'string',
                'description': 'Agent ID to create (for create action)',
            },
            'instance_id': {
                'type': 'string',
                'description': 'Agent instance ID (for terminate action)',
            },
        },
    },
    'search_hint': 'manage sub agents',
})


# ============== Exports ==============

def get_agent_tool() -> ToolRegistration:
    return AGENT_TOOL


def create_agent_manager(agents_dir: Optional[Path] = None) -> AgentManager:
    """创建 Agent 管理器"""
    return AgentManager(agents_dir)
