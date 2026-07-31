"""
Hello Agent — minimal Anvil agent that reads a file and answers a question.

Run:
    python examples/hello_agent/run.py

No API key needed; uses a mock LLM invoke to demonstrate the pattern.
Replace `invoke` with a real provider call to use with Claude/GPT/Gemini.
"""

from __future__ import annotations

import json
import os
import sys

from anvil.core.agent import AnvilAgent
from anvil.core.types import StopConfig
from anvil.steps.json_loop import JsonLoopState, make_json_decision_step


def mock_invoke(prompt: str) -> str:
    """Mock LLM that always returns the Anvil description.

    Replace this with a real provider:
        from anvil.llm.providers import anthropic_invoke_factory
        invoke = anthropic_invoke_factory(api_key=os.environ['ANTHROPIC_API_KEY'], model='claude-sonnet-5')
    """
    return json.dumps({
        'thought': 'I need to answer the question about Anvil.',
        'answer': 'Anvil is a lightweight Python agent framework for building LLM-powered coding assistants.',
        'done': True,
    })


def main() -> None:
    step = make_json_decision_step(mock_invoke, history_window=3)
    agent = AnvilAgent(
        step=step,
        stop=StopConfig(max_steps=5, max_elapsed_s=30.0),
    )

    result = agent.run(
        goal='What is Anvil? Give a one-sentence description.',
        initial_state=JsonLoopState(),
    )

    print(f'Done:        {result.done}')
    print(f'Stop reason: {result.stop_reason.value}')
    print(f'Steps:       {result.steps}')
    print(f'Answer:      {result.final_output}')


if __name__ == '__main__':
    main()
