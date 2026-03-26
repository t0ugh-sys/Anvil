# Repository Layout

## Top Level

- `skills/`: human-readable skill contracts and boundaries
- `docs/`: architecture and artifact references
- `examples/`: integration-oriented demos
- `src/loop_agent/`: runtime package
- `tests/`: unit and structural tests

## Runtime Package

- `core/`: generic loop engine and base types
- `llm/`: provider adapters and mock provider
- `memory/`: JSONL memory store and summary handling
- `steps/`: strategy-specific step builders
- `tools.py`: workspace-safe tool layer
- `task_graph.py`: dependency-aware task DAG state
- `mailbox.py`: persistent async message channel
- `subagents.py`: sub-agent runtime and dispatch helpers
- `policies.py`: capability-based permission governance
- `worktree_manager.py`: isolated task workspace manager
- `context_schema.py`: fixed compressed context payloads for agents
- `scheduler.py`: dependency-aware agent batch scheduler
- `coding_agent.py`: coding-agent orchestration
- `agent_cli.py`: coding-agent oriented CLI
- `cli.py`: general loop CLI
