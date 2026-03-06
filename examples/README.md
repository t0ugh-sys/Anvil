# LoopAgent Examples

This directory contains various examples for using LoopAgent.

## Quick Examples

| File | Description |
|------|-------------|
| `json_loop_stub_demo.py` | Basic JSON loop with mock responses |
| `logging_demo.py` | Using the logging system |
| `prompts_demo.py` | Using prompt templates |
| `browser_tools.py` | Browser automation with Playwright |

## Configuration Examples

### YAML Configuration (`config.yaml`)

```yaml
provider: openai_compatible
model: gpt-4o-mini
base_url: https://api.openai.com/v1
max_steps: 20
temperature: 0.2
history_window: 3
strategy: json_llm
skills:
  - web_search
  - memory
  - files
```

### Environment Variables (`.env`)

```bash
# Copy from .env.example
cp .env.example .env

# Edit .env with your API keys
```

## Running Examples

```bash
# Install package
pip install -e .

# Run JSON loop demo
python examples/json_loop_stub_demo.py

# Run with config
python -m loop_agent.agent_cli code --config config.yaml --goal "your goal"

# Use docker
docker build -t loopagent .
docker run loopagent --goal "your goal"
```

## Advanced Usage

### Custom Prompt Template

```python
from loop_agent.prompts import PromptTemplate, register_template

my_template = PromptTemplate(
    name="my_agent",
    template="Task: {{task}}\nContext: {{context}}",
    required_vars=["task", "context"],
)
register_template(my_template)
```

### Custom Skill

```python
from loop_agent.skills import Skill, register_skill

class MySkill(Skill):
    name = "my_skill"
    description = "Custom skill"
    
    def get_tools(self):
        def my_tool(args):
            return ToolResult(id="my_tool", ok=True, output="Done!", error=None)
        return {"my_tool": my_tool}

register_skill(MySkill)
```
