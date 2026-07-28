# Anvil

Terminal-first coding agent runtime. Core pattern:

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
python -m anvil.cli --goal "write hello world" --strategy demo --output json
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

- Python 3.10+
- Node 18+ (for npm wrapper)

## Usage

### Interactive mode

```bash
anvil                           # start interactive session
anvil --session-id <id>         # resume session
```

Slash commands: `/help`, `/status`, `/history`, `/todo`, `/tools`, `/exit`

### Batch mode

```bash
anvil code --goal "task description" --workspace . --provider anthropic --model claude-3-opus-20240229
```

### With skills

```bash
anvil code --goal "search for info" --skill web_search --skill memory
```

Built-in skills: `web_search`, `memory`, `files`, `commands`, `browser` (requires `playwright`)

## Provider Configuration

### Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-xxx
python -m anvil.cli --goal "task" --strategy json_llm --provider anthropic --model claude-3-opus-20240229
```

### OpenAI

```bash
export OPENAI_API_KEY=sk-xxx
python -m anvil.cli --goal "task" --strategy json_llm --provider openai_compatible --model gpt-4o-mini
```

### Gemini

```bash
export GEMINI_API_KEY=xxx
python -m anvil.cli --goal "task" --strategy json_llm --provider gemini --model gemini-pro
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
