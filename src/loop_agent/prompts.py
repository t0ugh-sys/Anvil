"""
Prompt Template System for LoopAgent

Provides reusable prompt templates with variable substitution.
"""

from __future__ import annotations

import re
from pathlib import Path


class PromptTemplate:
    """A prompt template with variable substitution."""
    
    def __init__(
        self,
        template: str,
        name: str = "default",
        description: str = "",
        required_vars: list[str] | None = None,
    ):
        self.template = template
        self.name = name
        self.description = description
        self.required_vars = required_vars or []
        
        # Find all variables in template
        self.variables = set(re.findall(r'\{\{(\w+)\}\}', template))
    
    def render(self, **kwargs: str) -> str:
        """Render the template with provided variables."""
        # Check required variables
        required_set = set(self.required_vars)
        missing = required_set - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        # Substitute variables
        result = self.template
        for key, value in kwargs.items():
            result = result.replace(f'{{{{{key}}}}}', str(value))
        
        return result
    
    def validate(self) -> bool:
        """Validate the template."""
        return True


# Built-in templates
DEFAULT_TEMPLATES: dict[str, PromptTemplate] = {}


def register_template(template: PromptTemplate) -> None:
    """Register a prompt template."""
    DEFAULT_TEMPLATES[template.name] = template


def get_template(name: str) -> PromptTemplate | None:
    """Get a template by name."""
    return DEFAULT_TEMPLATES.get(name)


def list_templates() -> list[str]:
    """List all registered templates."""
    return list(DEFAULT_TEMPLATES.keys())


# ============== Built-in Templates ==============

# Simple JSON decision template
register_template(PromptTemplate(
    name="json_loop",
    description="Standard JSON loop prompt",
    template="""Goal: {{goal}}

Context Summary:
{{state_summary}}

Recent Steps:
{{last_steps}}

Respond with JSON:
{
  "thought": "your reasoning",
  "plan": ["action 1", "action 2"],
  "tool_calls": [{"id": "call_1", "name": "tool_name", "arguments": {"arg": "value"}}],
  "final": null or "final answer"
}""",
    required_vars=["goal", "state_summary", "last_steps"],
))

# Coding agent template
register_template(PromptTemplate(
    name="coding",
    description="Coding agent prompt",
    template="""You are a coding assistant.

Goal: {{goal}}

Available tools:
- read_file: Read file contents
- write_file: Write content to file
- search: Search for patterns in files
- run_command: Execute shell commands

Workspace: {{workspace}}

History:
{{history}}

Tool Results:
{{tool_results}}

Think step by step. Use tools to accomplish the goal.
When done, respond with:
{
  "thought": "what you did",
  "plan": ["remaining steps"],
  "tool_calls": [...],
  "final": null or "completed: description"
}""",
    required_vars=["goal", "workspace", "history", "tool_results"],
))

# Analysis template
register_template(PromptTemplate(
    name="analyze",
    description="Analysis and reasoning prompt",
    template="""Analyze the following:

Goal: {{goal}}

Context:
{{context}}

Provide a thorough analysis and recommend actions.
Respond in JSON:
{
  "thought": "your analysis",
  "plan": ["recommendations"],
  "final": "conclusion or null if more steps needed"
}""",
    required_vars=["goal", "context"],
))

# Research template
register_template(PromptTemplate(
    name="research",
    description="Web research prompt",
    template="""Research task: {{goal}}

Use web_search to find information, then provide a summary.

Steps:
1. Search for relevant information
2. Fetch key pages
3. Synthesize findings

Respond:
{
  "thought": "what you found",
  "plan": ["next search or final answer"],
  "final": "summary or null"
}""",
    required_vars=["goal"],
))


# ============== Template Loader ==============

def load_templates_from_file(path: str | Path) -> dict[str, PromptTemplate]:
    """Load templates from a YAML or JSON file."""
    import json
    
    path = Path(path)
    suffix = path.suffix.lower()
    
    templates = {}
    
    if suffix in ('.yaml', '.yml'):
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
    elif suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported template file format: {suffix}")
    
    for name, config in data.items():
        if isinstance(config, str):
            template = PromptTemplate(template=config, name=name)
        elif isinstance(config, dict):
            template = PromptTemplate(
                template=config.get('template', ''),
                name=name,
                description=config.get('description', ''),
                required_vars=config.get('required_vars'),
            )
        else:
            continue  # Skip invalid config
        templates[name] = template
    
    return templates


def merge_templates(*sources: dict[str, PromptTemplate]) -> dict[str, PromptTemplate]:
    """Merge multiple template dictionaries."""
    result = DEFAULT_TEMPLATES.copy()
    for source in sources:
        result.update(source)
    return result
