from __future__ import annotations

from ..skills import get_skill, list_skills
from ..tools import build_default_tools, builtin_tool_specs

__all__ = ['render_tools', 'render_skills']


def render_tools(*, verbose: bool = False) -> str:
    specs = sorted(builtin_tool_specs(), key=lambda item: item.name)
    if verbose:
        lines: list[str] = []
        for item in specs:
            capabilities = ','.join(cap.value for cap in item.capabilities) or 'none'
            lines.append(f'{item.name}: {item.description} [{capabilities}] risk={item.risk_level.value}')
        return '\n'.join(lines)
    names = sorted(build_default_tools().keys())
    return '\n'.join(names)


def render_skills() -> str:
    lines = ['Available skills:']
    for name in list_skills():
        skill = get_skill(name)
        if skill:
            lines.append(f'  - {name}: {skill.description}')
    lines.append('')
    lines.append('Use --skill <name> to load specific skills')
    lines.append('Use --skill all to load all skills')
    return '\n'.join(lines)
