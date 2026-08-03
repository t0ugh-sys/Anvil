# Repository Layout

## Top Level

- `anvil/`: runtime package (moved from `src/anvil/` in Phase 1)
- `skills/`: human-readable skill contracts and boundaries
- `docs/`: architecture and artifact references
- `examples/`: integration-oriented demos
- `tests/`: unit and structural tests
- `docker/`: containerization configs
- `bin/`: executable scripts

## Runtime Package (`anvil/`)

### Core Modules
- `core/`: generic loop engine and base types
- `llm/`: provider adapters (Anthropic, OpenAI, Gemini, mock)
- `memory/`: JSONL memory store and summary handling
- `steps/`: strategy-specific step builders
- `tools/`: workspace-safe tool implementations
- `ops/`: high-level operations
- `ui/`: terminal UI components (rich_chat, tui_chat)
- `entrypoints/`: CLI and API entry points
- `services/`: service layer orchestration
- `commands/`: command implementations

### Internal Subpackages (Phase 2)

#### `infra/` — Infrastructure & Cross-Cutting Concerns
- `hooks.py`: pre/post execution hooks, security monitoring
- `skills.py`: skill discovery and loading
- `permissions.py`: permission manager and rules
- `policies.py`: capability-based governance
- `retry.py`: retry logic with exponential backoff
- `logging.py`: structured logging setup

#### `config/` — Configuration Management
- `loader.py`: environment config loading (formerly `config.py`)
- `layered.py`: layered config with merge semantics (formerly `layered_config.py`)
- `context_schema.py`: compressed context payloads for agents
- `run_schema.py`: run configuration schemas

#### `agent/` — Agent Orchestration
- `protocol.py`: agent protocol types (formerly `agent_protocol.py`)
- `loop.py`: tool-use execution loop (formerly `tool_use_loop.py`)
- `subagents.py`: sub-agent runtime and dispatch
- `background.py`: background agent execution

#### `runtime/` — Session & Task Runtime
- `code.py`: main CodeRuntime orchestration (formerly `runtime.py`)
- `team.py`: team runtime coordination (formerly `team_runtime.py`)
- `session.py`: session state management
- `mailbox.py`: persistent async message channel
- `scheduler.py`: dependency-aware agent batch scheduler
- `task_graph.py`: dependency-aware task DAG state
- `task_store.py`: task persistence
- `run_recorder.py`: run artifact recording

#### `compression/` — Context Compression
- `engine.py`: context compression engine (formerly `compression.py`, then `core.py` — renamed to avoid ambiguity with `anvil/core/`)

### Standalone Modules (Top-Level in `anvil/`)
- `api.py`: API interfaces
- `coding_agent.py`: coding-agent orchestration
- `messages.py`: message types
- `prompts.py`: prompt templates
- `errors.py`: error types
- `todo.py`: task tracking
- `token_estimation.py`: token counting utilities
- `tool_spec.py`: tool specification helpers
- `utils.py`: shared utilities
- `worktree_manager.py`: isolated task workspace manager

## Backward Compatibility

Shim files at old paths (e.g., `anvil/hooks.py` → `from anvil.infra.hooks import *`) maintain backward compatibility for existing imports during the transition period.
