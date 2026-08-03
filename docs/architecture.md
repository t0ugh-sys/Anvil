# Architecture

## Data Flow

```
User input
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Session / Chat Runtime                                  │
│  anvil/services/session_runtime.py                      │
│  anvil/services/chat_runtime.py                         │
└────────────────────────┬────────────────────────────────┘
                         │ messages + tools
                         ▼
┌─────────────────────────────────────────────────────────┐
│  LLM Provider                                           │
│  anvil/llm/anthropic/  │  anvil/llm/gemini.py          │
│  anvil/llm/openai_compat.py  │  anvil/llm/mock.py      │
└────────────────────────┬────────────────────────────────┘
                         │ response (text + tool_use blocks)
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Tool-Use Loop                                          │
│  anvil/agent/loop.py  –  execute_tool_use_round()       │
│                                                         │
│  while model calls tools:                              │
│    1. parse tool_use blocks                             │
│    2. dispatch_tool_calls() — parallel execution        │
│    3. append tool_result blocks                         │
│    4. call LLM again                                    │
└────────────────────────┬────────────────────────────────┘
                         │ tool calls
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Tool Layer                                             │
│  anvil/tools/__init__.py  –  execute_tool_call()        │
│                                                         │
│  read_file  write_file  apply_patch  run_command        │
│  search_files  list_dir  memory  git  github_cli        │
│  browser  web_search  todo                              │
└────────────────────────┬────────────────────────────────┘
                         │ accumulating messages (growing context)
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Context Compression                                    │
│  anvil/compression/engine.py  –  micro_compact_entries()│
│                                                         │
│  - Fires when context > 75% of model window             │
│  - Keeps last N tool results (configurable)             │
│  - Summarizes earlier history via a separate LLM call   │
└─────────────────────────────────────────────────────────┘
                         │ reply text
                         ▼
User output + run artifacts (.anvil/runs/<id>/)
```

## Key Modules

| Module | Role |
|--------|------|
| `anvil/core/agent.py` | `AnvilAgent` — main execution loop wrapper |
| `anvil/core/types.py` | `StopConfig`, `RunResult`, shared types |
| `anvil/agent/loop.py` | Tool-use round execution, `_ReadOnlyToolCache` |
| `anvil/agent/protocol.py` | `ToolCall`, `ToolResult`, step protocol types |
| `anvil/compression/engine.py` | `CompactConfig`, `micro_compact_entries` |
| `anvil/llm/providers.py` | Unified provider re-export |
| `anvil/services/session_runtime.py` | Interactive `anvil` CLI runtime |
| `anvil/services/coding_runtime.py` | `build_coding_decider` factory |
| `anvil/runtime/team.py` | `PersistentTeamRuntime` — multi-agent coordination |
| `anvil/tools/` | Tool implementations (12 built-ins) |

## Layered Configuration

Config is resolved in priority order (highest first):

```
CLI flags
  └── environment variables (.env)
        └── project config (.anvil/config.yaml)
              └── user config (~/.anvil/config.yaml)
                    └── defaults
```

Implementation: `anvil/services/session_runtime.py` → `build_layered_config()`

## Multi-Agent

```
PersistentTeamRuntime
  ├── Leader agent        (orchestrates via TeamMessage)
  └── Worker agent(s)     (each runs its own tool-use loop)
        ├── JsonlTeamInboxStore  (persistent mailbox)
        └── TaskGraph            (dependency tracking)
```

See [team.md](team.md) for usage details.

## Context Window Budget

Token usage is tracked per-call through `TokenUsageTracker` (in `anvil/llm/usage.py`).
Compression triggers at 75% of the model's context window (`_COMPACT_FRACTION = 0.75`).
Use `CompactConfig.for_model(model_id)` to get the right threshold automatically.
