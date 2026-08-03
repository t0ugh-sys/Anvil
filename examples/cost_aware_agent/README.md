# Cost-Aware Agent

An `AnvilAgent` with hard resource limits using `StopConfig` and a call-counting wrapper.

## What it demonstrates

- Using `StopConfig` with `max_steps` and `max_elapsed_s`
- Wrapping an LLM invoke to count and cap API calls
- Graceful early termination when a budget is reached

## Running

```bash
python examples/cost_aware_agent/run.py
```

No API key needed — uses a mock invoke.

## Key pattern

```python
stop = StopConfig(
    max_steps=10,       # hard step ceiling (independent of LLM)
    max_elapsed_s=30.0, # wall-clock timeout in seconds
)
agent = AnvilAgent(step=step, stop=stop)
```

`StopConfig` enforces limits regardless of what the LLM returns — the agent
stops as soon as any condition is met.

## Adding a real token budget

```python
from anvil.llm.usage import TokenUsageTracker
from anvil.llm.providers import anthropic_invoke_factory

tracker = TokenUsageTracker()
invoke = anthropic_invoke_factory(
    api_key=os.environ['ANTHROPIC_API_KEY'],
    model='claude-sonnet-5',
    usage_tracker=tracker,
)

# Check budget after each step via observer
def budget_observer(event):
    if tracker.total_output_tokens > 50_000:
        raise SystemExit('Token budget exceeded')
```

## Expected output

```
Done:        True
Stop reason: done_by_agent
Steps used:  3  (max 10)
LLM calls:   3  (budget 3)
Answer:      Budget reached — stopping early.
```
