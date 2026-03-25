from __future__ import annotations

import shutil
from pathlib import Path

from loop_agent.memory.jsonl_store import JsonlMemoryStore


def main() -> None:
    run_dir = Path('.loopagent/demo-memory')
    shutil.rmtree(run_dir, ignore_errors=True)

    store = JsonlMemoryStore(memory_dir=run_dir, summarize_every=2)
    store.on_event('run_started', {'goal': 'retain important user constraints', 'facts': ['prefer local execution']})
    store.on_event('step_started', {'step': 0, 'plan': ['read context', 'update summary']})
    store.on_event('step_succeeded', {'step': 0, 'output': 'read existing constraints'})
    store.on_event('run_finished', {'done': True, 'stop_reason': 'done', 'steps': 1})

    context = store.load_context(goal='retain important user constraints', last_k_steps=3)
    print(f'run_dir={run_dir}')
    print(f'summary_steps={context.state_summary.get("steps")}')
    print(f'facts={context.state_summary.get("facts")}')
    print(f'last_steps={list(context.last_steps)}')


if __name__ == '__main__':
    main()
