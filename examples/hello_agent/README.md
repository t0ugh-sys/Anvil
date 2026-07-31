# Hello Agent

Minimal Anvil agent that reads a file and answers a question.

## What it demonstrates

- Creating an `AnvilAgent` with a JSON decision loop
- Using `StopConfig` to limit execution
- Running an agent with a goal
- Mock LLM invoke (replace with real provider)

## Running

```bash
python examples/hello_agent/run.py
```

No API key needed — uses a mock LLM to demonstrate the pattern.

## Using a real provider

Replace the `mock_invoke` function with a real provider:

```python
from anvil.llm.providers import anthropic_invoke_factory

invoke = anthropic_invoke_factory(
    api_key=os.environ['ANTHROPIC_API_KEY'],
    model='claude-sonnet-5',
)

step = make_json_decision_step(invoke, history_window=3)
```

## Expected output

```
Done:        True
Stop reason: done_by_agent
Steps:       1
Answer:      Anvil is a lightweight Python agent framework for building LLM-powered coding assistants.
```
