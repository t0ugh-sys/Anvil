# Anvil

Anvil is a terminal-first coding agent runtime shaped more like Claude Code. The core pattern is:

```text
while model_is_calling_tools:
    response = LLM(messages, tools)
    execute tool calls
    append tool results
```

Everything else in the project layers on top of that loop: session runtime, permissions,
commands, memory, task graphs, subagents, worktree isolation, and scheduling.

The loop itself does not change when tools grow. We only extend the tool array and
dispatch map:

```text
+----------+      +-------+      +------------------+
|   User   | ---> |  LLM  | ---> | Tool Dispatch    |
|  prompt  |      |       |      | {                |
+----------+      +---+---+      |   bash: run_bash |
                     ^          |   read: run_read |
                     |          |   write: run_wr  |
                     +----------+   edit: run_edit |
                     tool_result| }                |
                                +------------------+
```

Key rule: the loop stays stable; the runtime layers around it can evolve independently.

## Highlights

- Terminal-first interactive runtime by default
- Claude Code-style internal layering: `entrypoints/`, `services/`, `commands/`
- Tool-use feedback loop as the primary execution model
- Stdlib-only core in `src/anvil/`
- Iterative agent loop with max-step, timeout, and cancellation stop conditions
- Structured run artifacts in `.anvil/runs/<run_id>/`
- Configurable model providers for mock, OpenAI-compatible, Anthropic, and Gemini flows
- Built-in tools for files, commands, memory analysis, git, and GitHub CLI workflows
- `unittest`-based test suite that runs without pytest

## Version Requirements

- Python library/runtime: Python 3.10+
- Node wrapper (`anvil` from npm): Python 3.11+ available on `PATH`
- Node.js: 18+

The npm wrapper bootstraps a local virtual environment under `~/.anvil/npm-bridge/` and installs the Python package there.

## Quick Start

### Python users

```bash
python -m pip install -e .
python -m unittest discover -s tests -p "test_*.py" -v
python -m anvil.cli --goal "write a one-line self introduction" --strategy demo --output json
```

### Node users

```bash
npm i -g git+https://github.com/t0ugh-sys/Anvil.git
anvil
anvil --session-id <session_id>
anvil tools
anvil code --goal "inspect README then finish" --workspace . --provider mock --model mock-v3 --output json
```

If you have multiple Python installations, set `ANVIL_PYTHON` before the first npm-backed run.

## Common Commands

### Core CLI

```bash
python -m anvil.cli --goal "answer in json protocol" --strategy json_stub --history-window 2
python -m anvil.cli --goal-file goal.txt --output json --include-history
python -m anvil.cli --goal-file goal.txt --observer-file events.jsonl
python -m anvil.cli --goal-file goal.txt --strategy json_stub --max-steps 1 --exit-on-failure
```

### Agent CLI

```bash
anvil
anvil --session-id <session_id>
anvil tools
anvil code --goal "inspect README then finish" --workspace . --provider mock --model mock-v3 --output json
```

The default `anvil` entrypoint starts an interactive session-first runtime. It persists chat state under `.anvil/sessions/<session_id>/` and supports basic slash commands:

```text
/help
/status
/config
/provider anthropic
/model claude-3-opus-20240229
/wire-api responses
/base-url https://api.openai.com/v1
/tools
/resume
/exit
```

You can switch provider and model during the same interactive session. The current runtime config is persisted in the session and restored on resume.
Use `/status` to inspect the current session, todo state, permission counts, and recent tool activity.
Use `/tools` to inspect the current tool surface grouped by capability and risk.
Interactive startup now shows a session header with turn, message, command, and step counts.

`anvil code ...` remains the explicit non-interactive batch runtime.

The `anvil code` entrypoint is still the direct CLI surface of the core loop:

```text
model decides -> tool calls execute -> tool results feed back -> stop or continue
```

Useful first commands:

```bash
anvil --help
anvil code --help
anvil tools
anvil --provider anthropic --model claude-3-opus-20240229
anvil replay --session-id <session_id> --pretty --limit 20
anvil replay --session-id <session_id> --pretty --filter permissions-only
```

## Runtime Layout

The codebase is now organized more like a terminal coding product runtime than a flat loop library:

- `src/anvil/entrypoints/`: top-level command entrypoints
- `src/anvil/services/`: session, chat, and coding runtime orchestration
- `src/anvil/commands/`: slash-command parsing and execution
- `src/anvil/tools.py` and tool metadata modules: tool dispatch layer
- `src/anvil/core/`: lower-level reusable agent primitives

## Visible Progress

Anvil can keep a visible todo list inside the same tool-use loop.

- The model updates progress through the `todo_write` tool
- The runtime stores todo state in `ToolUseState`
- The current todo list is injected back into `state_summary.todo_state`
- If open todos are not updated for 3 rounds, the runtime injects `todo_reminder`

Minimal tool payload:

```json
{
  "thought": "track progress",
  "plan": ["inspect repo", "edit file"],
  "tool_calls": [
    {
      "id": "call_1",
      "name": "todo_write",
      "arguments": {
        "items": [
          {"id": "t1", "content": "inspect repo", "status": "completed"},
          {"id": "t2", "content": "edit file", "status": "in_progress"}
        ]
      }
    }
  ],
  "final": null
}
```

Typical event metadata now includes:

```text
todo_state.items
todo_state.lines
todo_state.rounds_since_update
todo_reminder
```

### Conda example

```powershell
$env:PYTHONPATH="src"
conda --no-plugins run --no-capture-output -n base python -m unittest discover -s tests -p "test_*.py" -v
conda --no-plugins run --no-capture-output -n base python -m anvil.cli --goal-file .\goal.txt --strategy demo
```

## Strategies

- `demo`: fixed multi-step convergence demo
- `json_stub`: JSON protocol loop with stubbed model output
- `json_llm`: JSON protocol loop backed by a configured provider/model

Example:

```bash
python -m anvil.cli --goal "answer in json protocol" --strategy json_llm --provider mock --model qwen-max
```

## Providers

Pick a provider and model at startup:

```bash
anvil --provider anthropic --model claude-3-opus-20240229
python -m anvil.cli --goal "answer in json protocol" --strategy json_llm --provider openai_compatible --model gpt-4o-mini --base-url https://api.openai.com/v1 --wire-api chat_completions
python -m anvil.cli --goal "answer in json protocol" --strategy json_llm --provider openai_compatible --model gpt-5.3-codex --base-url https://codex-api.packycode.com/v1 --wire-api responses
```

Switch runtime config inside an interactive session:

```text
/config
/provider anthropic
/model claude-3-opus-20240229
/wire-api responses
/base-url https://api.openai.com/v1
```

Extra provider headers, retries, and fallback models are still available through the `code` command flags when needed.

## Run Recording and Memory

Anvil records run data by default.

- Run records: `runs/<timestamp>/`
- Memory root: `.anvil/runs/<run_id>/`
- Event stream: `events.jsonl`
- Snapshot state: `state.json`
- Summary: `summary.json`

Useful flags:

- `--no-record-run`
- `--runs-dir`
- `--memory-dir`
- `--run-id`
- `--summarize-every`
- `--history-window`

Each iteration can inject structured memory context:

- `state_summary`
- `last_steps`

Example `goal.txt` content should be saved as UTF-8:

```text
Write a self introduction in under 50 Chinese characters.
```

## Skills

Built-in skills:

| Skill | Purpose | Extra dependency |
| --- | --- | --- |
| `web_search` | Search the web and fetch pages | none |
| `memory` | Analyze past runs | none |
| `files` | Read, write, patch, and search files | none |
| `commands` | Run shell commands | none |
| `browser` | Browser automation | `playwright` |

Anvil uses two-layer skill injection:

- Layer 1: only skill `name + description` goes into the prompt
- Layer 2: full skill instructions are loaded on demand through `load_skill`

Load specific skills:

```bash
anvil code --goal "search for info" --skill web_search --skill memory
```

## Project Layout

- `skills/`: built-in skill notes and extension references
- `skills/<name>/SKILL.md`: skill frontmatter plus on-demand full instructions
- `src/anvil/`: core package
- `src/anvil/core/`: generic loop engine, stop rules, and serialization
- `src/anvil/steps/`: reusable step strategies and registries
- `src/anvil/memory/`: JSONL memory store and context loading
- `src/anvil/llm/`: provider adapters
- `src/anvil/ops/`: provider doctor, git, and GitHub operational helpers
- `src/anvil/tool_use_loop.py`: the central model -> tools -> results loop
- `src/anvil/tools.py`: builtin tool registry, dispatch map, and execution boundary
- `src/anvil/ui/`: optional chat-oriented TUI entrypoints
- `tests/`: unit tests
- `examples/`: optional demos and integrations
- `bin/anvil.js`: npm bridge entrypoint for `anvil`
- `.github/workflows/tests.yml`: Python test workflow
- `.github/workflows/release.yml`: npm release workflow

## Development

```bash
python -m pip install -e .
python -m unittest discover -s tests -p "test_*.py" -v
npm pack --dry-run
```

## Notes

- The core engine returns `stop_reason=step_error` instead of crashing the process when a step raises.
- `requirements.txt` is only used as an examples/extension placeholder.
- `npm i -g @t0ugh-sys/anvil` works only after the package has been published to npm.
