"""
SkillTool - Dynamic Skill Loading

参考 Claude Code SkillTool 的设计，支持动态技能加载。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loop_agent.agent_protocol import ToolResult
from loop_agent.tool_def import ToolRegistration, ToolUseContext
from loop_agent.tool_builder import build_tool


# ============== Skill Definition ==============

@dataclass
class Skill:
    """Skill 定义"""
    name: str
    description: str
    prompt: str = ''
    tools: List[str] = field(default_factory=list)
    commands: List[str] = field(default_factory=list)


# ============== Skill Loader ==============

class SkillLoader:
    """动态技能加载器"""
    
    def __init__(self, skills_dir: Optional[Path] = None):
        self._loaded: Dict[str, Skill] = {}
        self._skills_dir = skills_dir
    
    def load(self, name: str) -> bool:
        """加载技能"""
        # 先尝试从已注册技能加载
        from ..skills import get_skill
        
        skill = get_skill(name)
        if skill:
            self._loaded[name] = Skill(
                name=skill.name,
                description=skill.description,
                prompt=skill.get_body(),
            )
            return True
        
        return False
    
    def unload(self, name: str) -> bool:
        """卸载技能"""
        if name in self._loaded:
            del self._loaded[name]
            return True
        return False
    
    def is_loaded(self, name: str) -> bool:
        """检查技能是否已加载"""
        return name in self._loaded
    
    def list_loaded(self) -> List[str]:
        """列出已加载的技能"""
        return list(self._loaded.keys())
    
    def get_skill(self, name: str) -> Optional[Skill]:
        """获取技能"""
        return self._loaded.get(name)
    
    def get_prompt(self, name: str) -> Optional[str]:
        """获取技能提示"""
        skill = self._loaded.get(name)
        return skill.prompt if skill else None


# ============== SkillTool Handler ==============

def _skill_handler(
    context: ToolUseContext,
    args: Dict[str, Any],
) -> ToolResult:
    """Skill 工具处理函数"""
    call_id = str(args.get('id', 'skill'))
    
    action = str(args.get('action', 'list')).strip().lower()
    
    # 获取技能加载器
    skill_loader = context.skill_loader
    if skill_loader is None:
        return ToolResult(
            id=call_id,
            ok=False,
            output='',
            error='skill_loader not configured',
        )
    
    try:
        if action == 'list':
            # 列出已加载的技能
            loaded = skill_loader.list_loaded()
            lines = ['Loaded skills:']
            
            if not loaded:
                lines.append('  (none)')
            else:
                for name in loaded:
                    skill = skill_loader.get_skill(name)
                    if skill:
                        lines.append(f'  - {name}: {skill.description}')
            
            return ToolResult(id=call_id, ok=True, output='\n'.join(lines), error=None)
        
        elif action == 'load':
            # 加载技能
            name = str(args.get('name', '')).strip()
            if not name:
                return ToolResult(id=call_id, ok=False, output='', error='name is required')
            
            success = skill_loader.load(name)
            if success:
                return ToolResult(id=call_id, ok=True, output=f'Loaded: {name}', error=None)
            else:
                return ToolResult(id=call_id, ok=False, output='', error=f'Failed to load: {name}')
        
        elif action == 'unload':
            # 卸载技能
            name = str(args.get('name', '')).strip()
            if not name:
                return ToolResult(id=call_id, ok=False, output='', error='name is required')
            
            success = skill_loader.unload(name)
            if success:
                return ToolResult(id=call_id, ok=True, output=f'Unloaded: {name}', error=None)
            else:
                return ToolResult(id=call_id, ok=False, output='', error=f'Not loaded: {name}')
        
        elif action == 'info':
            # 获取技能信息
            name = str(args.get('name', '')).strip()
            if not name:
                return ToolResult(id=call_id, ok=False, output='', error='name is required')
            
            prompt = skill_loader.get_prompt(name)
            if prompt:
                return ToolResult(id=call_id, ok=True, output=prompt, error=None)
            else:
                return ToolResult(id=call_id, ok=False, output='', error=f'Not loaded: {name}')
        
        else:
            return ToolResult(id=call_id, ok=False, output='', error=f'Unknown action: {action}')
    
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


# ============== Tool Registration ==============

SKILL_TOOL = build_tool({
    'name': 'skill',
    'description': 'Load, unload, and manage skills',
    'handler': _skill_handler,
    'input_schema': {
        'type': 'object',
        'properties': {
            'action': {
                'type': 'string',
                'description': 'Action: list, load, unload, info',
                'enum': ['list', 'load', 'unload', 'info'],
            },
            'name': {
                'type': 'string',
                'description': 'Skill name (for load, unload, info actions)',
            },
        },
    },
    'search_hint': 'manage dynamic skills',
})


# ============== Exports ==============

def get_skill_tool() -> ToolRegistration:
    return SKILL_TOOL
