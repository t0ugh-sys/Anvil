"""
Skill System for LoopAgent

Skills are pluggable capabilities that can be loaded dynamically.
Each skill provides tools and/or prompt enhancements for the agent.

Usage:
    from loop_agent.skills import load_skill, build_skill_tools
    
    # Load a built-in skill
    load_skill('web_search')
    
    # Get tools from skill
    tools = build_skill_tools()
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from loop_agent.agent_protocol import ToolResult


# Skill definition
class Skill:
    """Base class for skills."""
    
    name: str
    description: str
    
    def get_tools(self) -> Dict[str, Callable]:
        """Return tools provided by this skill."""
        return {}
    
    def get_prompt_context(self) -> Dict[str, Any]:
        """Return additional context for prompts."""
        return {}


# Built-in skills registry
_SKILL_REGISTRY: Dict[str, type[Skill]] = {}


def register_skill(skill_class: type[Skill]) -> None:
    """Register a skill class."""
    _SKILL_REGISTRY[skill_class.name] = skill_class


def get_skill(name: str) -> Optional[Skill ]:
    """Get a skill instance by name."""
    skill_class = _SKILL_REGISTRY.get(name)
    if skill_class:
        return skill_class()
    return None


def list_skills() -> List[str]:
    """List all registered skills."""
    return list(_SKILL_REGISTRY.keys())


# ============== Built-in Skills ==============

class WebSearchSkill(Skill):
    """Web search and fetch capabilities."""
    name = "web_search"
    description = "Search the web and fetch URLs"
    
    def get_tools(self) -> Dict[str, Callable]:
        from loop_agent.tools import web_search_tool, fetch_url_tool
        return {
            'web_search': web_search_tool,
            'fetch_url': fetch_url_tool,
        }


class MemorySkill(Skill):
    """Memory and learning capabilities."""
    name = "memory"
    description = "Analyze past runs and learn patterns"
    
    def get_tools(self) -> Dict[str, Callable]:
        from loop_agent.tools import analyze_memory_tool
        return {
            'analyze_memory': analyze_memory_tool,
        }


class FileSkill(Skill):
    """File operations."""
    name = "files"
    description = "Read, write, and search files"
    
    def get_tools(self) -> Dict[str, Callable]:
        from loop_agent.tools import read_file_tool, write_file_tool, apply_patch_tool, search_tool
        return {
            'read_file': read_file_tool,
            'write_file': write_file_tool,
            'apply_patch': apply_patch_tool,
            'search': search_tool,
        }


class CommandSkill(Skill):
    """Shell command execution."""
    name = "commands"
    description = "Run shell commands"
    
    def get_tools(self) -> Dict[str, Callable]:
        from loop_agent.tools import run_command_tool
        return {
            'run_command': run_command_tool,
        }


class BrowserSkill(Skill):
    """Browser automation using Playwright.
    
    Requires: pip install playwright
    """
    name = "browser"
    description = "Browser automation (requires playwright)"
    
    def get_tools(self) -> Dict[str, Callable]:
        # Lazy import - only load when skill is used
        try:
            from loop_agent.skills.browser_tools import get_browser_tools
            return get_browser_tools()
        except ImportError:
            # Return a error tool if playwright not installed
            def browser_not_available(args):
                return ToolResult(
                    id=str(args.get('id', 'browser')),
                    ok=False,
                    output='',
                    error="Browser skill requires playwright: pip install playwright"
                )
            return {'browser': browser_not_available}


# Register built-in skills
register_skill(WebSearchSkill)
register_skill(MemorySkill)
register_skill(FileSkill)
register_skill(CommandSkill)


# ============== Skill Loading API ==============

class SkillLoader:
    """Dynamic skill loader for external skills."""
    
    def __init__(self):
        self._loaded_skills: Dict[str, Skill] = {}
    
    def load(self, name: str) -> bool:
        """Load a skill by name."""
        # Check built-in skills
        skill = get_skill(name)
        if skill:
            self._loaded_skills[name] = skill
            return True
        
        # Try loading external skill
        return self._load_external(name)
    
    def _load_external(self, name: str) -> bool:
        """Try to load an external skill."""
        try:
            # Try importing as a module
            import importlib
            module = importlib.import_module(f'loopagent_skills.{name}')
            
            if hasattr(module, 'Skill'):
                skill = module.Skill()
                self._loaded_skills[name] = skill
                return True
        except ImportError:
            pass
        return False
    
    def unload(self, name: str) -> bool:
        """Unload a skill."""
        if name in self._loaded_skills:
            del self._loaded_skills[name]
            return True
        return False
    
    def get_tools(self) -> Dict[str, Callable]:
        """Get all tools from loaded skills."""
        tools: Dict[str, Callable] = {}
        for skill in self._loaded_skills.values():
            tools.update(skill.get_tools())
        return tools
    
    def get_prompt_context(self) -> Dict[str, Any]:
        """Get combined prompt context from all loaded skills."""
        context: Dict[str, Any] = {}
        for skill in self._loaded_skills.values():
            ctx = skill.get_prompt_context()
            context.update(ctx)
        return context
    
    def list_loaded(self) -> List[str]:
        """List loaded skill names."""
        return list(self._loaded_skills.keys())


# Global skill loader instance
_default_loader = SkillLoader()


# Convenience functions
def load_skill(name: str) -> bool:
    """Load a skill by name."""
    return _default_loader.load(name)


def unload_skill(name: str) -> bool:
    """Unload a skill by name."""
    return _default_loader.unload(name)


def build_skill_tools() -> Dict[str, Callable]:
    """Get all tools from loaded skills."""
    return _default_loader.get_tools()


def get_prompt_context() -> Dict[str, Any]:
    """Get combined prompt context from all loaded skills."""
    return _default_loader.get_prompt_context()


def list_loaded_skills() -> List[str]:
    """List all loaded skill names."""
    return _default_loader.list_loaded()


# ============== CLI Integration ==============

def add_skill_arguments(parser: Any) -> None:
    """Add skill-related arguments to an argument parser."""
    parser.add_argument(
        '--skill',
        action='append',
        default=[],
        dest='skills',
        help='Skills to load (can be repeated)',
    )


def load_skills_from_args(args: Any) -> SkillLoader:
    """Load skills from parsed arguments."""
    loader = SkillLoader()
    
    skills_arg = getattr(args, 'skills', []) or []
    
    # Load default skills if none specified
    if not skills_arg:
        # Load all built-in skills by default
        for skill_name in list_skills():
            loader.load(skill_name)
    else:
        # Load specified skills
        for skill_name in skills_arg:
            if not loader.load(skill_name):
                raise ValueError(f'Unknown skill: {skill_name}')
    
    return loader
