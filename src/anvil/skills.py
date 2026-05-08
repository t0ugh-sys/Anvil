"""
Skill System for Anvil

Skills are pluggable capabilities that can be loaded dynamically.
Each skill provides tools and/or prompt enhancements for the agent.

Usage:
    from anvil.skills import load_skill, build_skill_tools
    
    # Load a built-in skill
    load_skill('web_search')
    
    # Get tools from skill
    tools = build_skill_tools()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .agent_protocol import ToolResult


# Skill definition
class Skill:
    """Base class for skills."""
    
    name: str
    description: str
    
    def get_tools(self) -> dict[str, Callable]:
        """Return tools provided by this skill."""
        return {}
    
    def get_prompt_context(self) -> dict[str, Any]:
        """Return additional context for prompts."""
        return {}

    def get_body(self) -> str:
        """Return full skill instructions for on-demand loading."""
        return ''


class LocalDocumentSkill(Skill):
    def __init__(self, *, name: str, description: str, body: str) -> None:
        self.name = name
        self.description = description
        self._body = body

    def get_body(self) -> str:
        return self._body


# Built-in skills registry
_SKILL_REGISTRY: dict[str, type[Skill]] = {}


def register_skill(skill_class: type[Skill]) -> None:
    """Register a skill class."""
    _SKILL_REGISTRY[skill_class.name] = skill_class


def _skills_docs_root() -> Path:
    return Path(__file__).resolve().parents[2] / 'skills'


def _skill_doc_path(name: str) -> Path:
    return _skills_docs_root() / name / 'SKILL.md'


def _legacy_skill_doc_path(name: str) -> Path:
    return _skills_docs_root() / f'{name}.md'


def _skill_doc_path_for_root(root: Path, name: str) -> Path:
    return root / name / 'SKILL.md'


def _legacy_skill_doc_path_for_root(root: Path, name: str) -> Path:
    return root / f'{name}.md'


def _parse_skill_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    text = raw.strip()
    if not text.startswith('---\n'):
        return {}, text

    lines = text.splitlines()
    meta: dict[str, str] = {}
    end_index = -1
    for index in range(1, len(lines)):
        line = lines[index].strip()
        if line == '---':
            end_index = index
            break
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        meta[key.strip()] = value.strip()

    if end_index < 0:
        return {}, text

    body = '\n'.join(lines[end_index + 1 :]).strip()
    return meta, body


def _read_skill_doc(name: str) -> tuple[dict[str, str], str]:
    path = _skill_doc_path(name)
    if path.exists():
        return _parse_skill_frontmatter(path.read_text(encoding='utf-8'))

    legacy_path = _legacy_skill_doc_path(name)
    if legacy_path.exists():
        return _parse_skill_frontmatter(legacy_path.read_text(encoding='utf-8'))

    return {}, ''


def _read_skill_doc_from_root(root: Path, name: str) -> tuple[dict[str, str], str]:
    path = _skill_doc_path_for_root(root, name)
    if path.exists():
        return _parse_skill_frontmatter(path.read_text(encoding='utf-8'))

    legacy_path = _legacy_skill_doc_path_for_root(root, name)
    if legacy_path.exists():
        return _parse_skill_frontmatter(legacy_path.read_text(encoding='utf-8'))

    return {}, ''


def discover_local_skill_names(docs_root: Path | None = None) -> list[str]:
    root = docs_root or _skills_docs_root()
    names = set()
    if not root.exists():
        return []

    for child in root.iterdir():
        if child.is_dir() and (child / 'SKILL.md').exists():
            names.add(child.name)
        elif child.is_file() and child.suffix.lower() == '.md' and child.name != 'README.md':
            names.add(child.stem)
    return sorted(names)


def _build_local_document_skill(name: str, docs_root: Path | None = None) -> Skill | None:
    root = docs_root or _skills_docs_root()
    meta, body = _read_skill_doc_from_root(root, name)
    if not body and not meta:
        return None
    skill_name = meta.get('name', name)
    description = meta.get('description', f'Local skill: {skill_name}')
    return LocalDocumentSkill(name=skill_name, description=description, body=body)


def get_skill(name: str, docs_root: Path | None = None) -> Skill | None:
    """Get a skill instance by name."""
    skill_class = _SKILL_REGISTRY.get(name)
    if skill_class:
        return skill_class()
    return _build_local_document_skill(name, docs_root=docs_root)


def list_skills() -> list[str]:
    """List all registered skills."""
    names = set(_SKILL_REGISTRY.keys())
    names.update(discover_local_skill_names())
    return sorted(names)


def skill_metadata(name: str) -> dict[str, str] | None:
    skill = get_skill(name)
    if skill is None:
        return None
    meta, _ = _read_skill_doc(name)
    return {
        'name': meta.get('name', skill.name),
        'description': meta.get('description', skill.description),
    }


# ============== Built-in Skills ==============

class WebSearchSkill(Skill):
    """Web search and fetch capabilities."""
    name = "web_search"
    description = "Search the web and fetch URLs"
    
    def get_tools(self) -> dict[str, Callable]:
        from .tools import web_search_tool, fetch_url_tool
        return {
            'web_search': web_search_tool,
            'fetch_url': fetch_url_tool,
        }


class MemorySkill(Skill):
    """Memory and learning capabilities."""
    name = "memory"
    description = "Analyze past runs and learn patterns"
    
    def get_tools(self) -> dict[str, Callable]:
        from .tools import analyze_memory_tool
        return {
            'analyze_memory': analyze_memory_tool,
        }


class FileSkill(Skill):
    """File operations."""
    name = "files"
    description = "Read, write, and search files"
    
    def get_tools(self) -> dict[str, Callable]:
        from .tools import read_file_tool, write_file_tool, apply_patch_tool, search_tool
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
    
    def get_tools(self) -> dict[str, Callable]:
        from .tools import run_command_tool
        return {
            'run_command': run_command_tool,
        }


class BrowserSkill(Skill):
    """Browser automation using Playwright.
    
    Requires: pip install playwright
    """
    name = "browser"
    description = "Browser automation (requires playwright)"
    
    def get_tools(self) -> dict[str, Callable]:
        # Lazy import - only load when skill is used
        try:
            from examples.browser_tools import build_browser_tools
            return build_browser_tools()
        except ImportError:
            # Return a error tool if playwright not installed
            def browser_not_available(context, args):
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
register_skill(BrowserSkill)


# ============== Skill Loading API ==============

class SkillLoader:
    """Dynamic skill loader for external skills."""
    
    def __init__(self, *, docs_root: Path | None = None):
        self._loaded_skills: dict[str, Skill] = {}
        self._docs_root = docs_root or _skills_docs_root()
    
    def load(self, name: str) -> bool:
        """Load a skill by name."""
        # Check built-in skills
        skill = get_skill(name, docs_root=self._docs_root)
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
            module = importlib.import_module(f'anvil_skills.{name}')
            
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
    
    def get_tools(self) -> dict[str, Callable]:
        """Get all tools from loaded skills."""
        tools: dict[str, Callable] = {}
        for skill in self._loaded_skills.values():
            tools.update(skill.get_tools())
        return tools
    
    def get_prompt_context(self) -> dict[str, Any]:
        """Get combined prompt context from all loaded skills."""
        context: dict[str, Any] = {}
        for skill in self._loaded_skills.values():
            ctx = skill.get_prompt_context()
            context.update(ctx)
        return context

    def metadata(self) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        for skill in self._loaded_skills.values():
            meta, _ = _read_skill_doc_from_root(self._docs_root, skill.name)
            items.append({
                'name': meta.get('name', skill.name),
                'description': meta.get('description', skill.description),
            })
        return items

    def load_body(self, name: str) -> str | None:
        skill = self._loaded_skills.get(name)
        if skill is None:
            return None
        meta, body = _read_skill_doc_from_root(self._docs_root, name)
        if body:
            return body
        generated = skill.get_body().strip()
        if generated:
            return generated
        title = meta.get('name', skill.name)
        description = meta.get('description', skill.description)
        return f'# {title}\n\n{description}'
    
    def list_loaded(self) -> list[str]:
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


def build_skill_tools() -> dict[str, Callable]:
    """Get all tools from loaded skills."""
    return _default_loader.get_tools()


def get_prompt_context() -> dict[str, Any]:
    """Get combined prompt context from all loaded skills."""
    return _default_loader.get_prompt_context()


def list_loaded_skills() -> list[str]:
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
