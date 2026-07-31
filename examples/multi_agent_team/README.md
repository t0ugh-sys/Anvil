# Multi-Agent Team

Two AnvilAgent instances collaborating: a Planner decomposes the goal, an Executor runs each step.

## What it demonstrates

- Coordinating multiple `AnvilAgent` instances
- Sequential planner → executor pipeline
- Parsing structured output from one agent to feed the next

## Running

```bash
python examples/multi_agent_team/run.py
```

No API key needed — uses mock LLM invokes.

## Using real providers

```python
from anvil.llm.providers import anthropic_invoke_factory

planner_invoke = anthropic_invoke_factory(
    api_key=os.environ['ANTHROPIC_API_KEY'],
    model='claude-opus-5',          # stronger model for planning
)
executor_invoke = anthropic_invoke_factory(
    api_key=os.environ['ANTHROPIC_API_KEY'],
    model='claude-haiku-4-5-20251001',  # fast model for execution
)
```

## Expected output

```
Goal: Refactor the legacy calculator module for readability and type safety

[Planner] Decomposing goal into steps...
  Plan (3 steps):
    1. Rename `calc` to `calculate`
    2. Extract `validate_input` helper
    3. Add type annotations

[Executor] Running each step...
  ✓ Done: Rename `calc` to `calculate`
  ✓ Done: Extract `validate_input` helper
  ✓ Done: Add type annotations

Team complete. 3/3 steps executed.
```

## Next steps

For production multi-agent systems with persistent state and inter-agent messaging,
see `anvil.agent.subagents` and `anvil.runtime.team.PersistentTeamRuntime`.
