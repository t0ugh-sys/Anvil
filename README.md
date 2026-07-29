# Anvil

Interactive coding agent with a conversational chat interface. Core pattern:

```python
while model_is_calling_tools:
    response = LLM(messages, tools)
    execute tool calls
    append tool results
```

Everything else layers on top: session runtime, permissions, commands, memory, task graphs, subagents, worktree isolation, scheduling.

## Highlights

- Terminal-first interactive runtime
- Tool-use feedback loop as primary execution model
- Stdlib-only core in `anvil/core/`
- Structured run artifacts in `.anvil/runs/<run_id>/`
- Multi-provider support: Anthropic, OpenAI, Gemini, mock
- Built-in tools: files, commands, memory, git, GitHub CLI
- `unittest`-based test suite (no pytest required)

## Quick Start

```bash
# Install
python -m pip install -e .

# Test
python -m unittest discover -s tests -p "test_*.py" -v

# Run
anvil
```

## Package Structure

```
anvil/
├── core/          loop engine, base types
├── llm/           provider adapters (Anthropic, OpenAI, Gemini, mock)
├── infra/         hooks, skills, permissions, policies, retry, logging
├── config/        configuration management
├── agent/         agent protocol, tool-use loop, subagents
├── runtime/       session, task graph, team runtime, mailbox
├── compression/   context compression
├── tools/         tool implementations
├── services/      chat/coding runtime orchestration
├── commands/      slash commands
├── entrypoints/   CLI entry points
├── memory/        JSONL memory store
├── steps/         strategy implementations
├── ops/           git/github helpers
└── ui/            TUI components
```

See [docs/repo-layout.md](docs/repo-layout.md) for full structure details.

## Requirements

- Python 3.10+ (3.11+ recommended)
- Node 18+ (for npm wrapper)

## Usage

```bash
anvil                           # start interactive session
anvil --session-id <id>         # resume session
```

Slash commands: `/help`, `/status`, `/history`, `/todo`, `/tools`, `/pricing`, `/exit`

Skills live in `skills/` at the repo root. Each skill follows the layout:

```
skills/<name>/SKILL.md    # skill manifest and instructions
```

Built-in skills: `web_search`, `memory`, `files`, `commands`, `browser` (requires `playwright`)

### Built-in tools

Key tools available to the agent: `todo_write`, `todo_reminder`, `run_command`, `read_file`, `write_file`, `search_files`.

## Provider Configuration

Set the API key for your provider, then launch the interactive UI — it will pick up the provider and model from your session config or `/status` settings.

### Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-xxx
anvil
```

### OpenAI

```bash
export OPENAI_API_KEY=sk-xxx
anvil
```

### Gemini

```bash
export GEMINI_API_KEY=xxx
anvil
```

## Run Recording

Anvil records runs by default to `.anvil/runs/<run_id>/`:

- `events.jsonl` — event stream
- `state.json` — snapshot state
- `summary.json` — run summary

Flags: `--no-record-run`, `--runs-dir`, `--memory-dir`, `--run-id`

## Development

```bash
python -m pip install -e .
python -m unittest discover -s tests -p "test_*.py" -v
```

## License

MIT
