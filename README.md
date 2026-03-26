# LoopAgent

LoopAgent is a tool-use feedback loop framework. The core pattern is:

```text
while model_is_calling_tools:
    response = LLM(messages, tools)
    execute tool calls
    append tool results
```

Everything else in the project layers on top of that loop: policy, memory, hooks,
task graphs, subagents, worktree isolation, and scheduling.

## Highlights

- Tool-use feedback loop as the primary runtime model
- Stdlib-only core in `src/loop_agent/`
- Iterative agent loop with max-step, timeout, and cancellation stop conditions
- Structured run artifacts in `.loopagent/runs/<run_id>/`
- Configurable model providers for mock, OpenAI-compatible, Anthropic, and Gemini flows
- Built-in tools for files, commands, memory analysis, git, and GitHub CLI workflows
- `unittest`-based test suite that runs without pytest

## Version Requirements

- Python library/runtime: Python 3.10+
- Node wrapper (`loopagent` from npm): Python 3.11+ available on `PATH`
- Node.js: 18+

The npm wrapper bootstraps a local virtual environment under `~/.loopagent/npm-bridge/` and installs the Python package there.

## Quick Start

### Python users

```bash
python -m pip install -e .
python -m unittest discover -s tests -p "test_*.py" -v
python -m loop_agent.cli --goal "write a one-line self introduction" --strategy demo --output json
```

### Node users

```bash
npm i -g git+https://github.com/t0ugh-sys/LoopAgent.git
loopagent tools
loopagent code --goal "inspect README then finish" --workspace . --provider mock --model mock-v3 --output json
```

If you have multiple Python installations, set `LOOPAGENT_PYTHON` before the first npm-backed run.

## Common Commands

### Core CLI

```bash
python -m loop_agent.cli --goal "answer in json protocol" --strategy json_stub --history-window 2
python -m loop_agent.cli --goal-file goal.txt --output json --include-history
python -m loop_agent.cli --goal-file goal.txt --observer-file events.jsonl
python -m loop_agent.cli --goal-file goal.txt --strategy json_stub --max-steps 1 --exit-on-failure
```

### Agent CLI

```bash
python -m loop_agent.agent_cli tools
python -m loop_agent.agent_cli code --goal "inspect README then finish" --workspace . --provider mock --model mock-v3 --output json
```

The `agent_cli` entrypoint is the direct CLI surface of the core loop:

```text
model decides -> tool calls execute -> tool results feed back -> stop or continue
```

Useful first commands:

```bash
python -m loop_agent.agent_cli --help
python -m loop_agent.agent_cli code --help
python -m loop_agent.agent_cli tools
```

### Conda example

```powershell
$env:PYTHONPATH="src"
conda --no-plugins run --no-capture-output -n base python -m unittest discover -s tests -p "test_*.py" -v
conda --no-plugins run --no-capture-output -n base python -m loop_agent.cli --goal-file .\goal.txt --strategy demo
```

## Strategies

- `demo`: fixed multi-step convergence demo
- `json_stub`: JSON protocol loop with stubbed model output
- `json_llm`: JSON protocol loop backed by a configured provider/model

Example:

```bash
python -m loop_agent.cli --goal "answer in json protocol" --strategy json_llm --provider mock --model qwen-max
```

## Provider Examples

### OpenAI-compatible

```bash
set OPENAI_API_KEY=sk-xxx
python -m loop_agent.cli --goal "answer in json protocol" --strategy json_llm --provider openai_compatible --model gpt-4o-mini --base-url https://api.openai.com/v1 --wire-api chat_completions
```

### Responses API

```bash
set OPENAI_API_KEY=sk-xxx
python -m loop_agent.cli --goal "answer in json protocol" --strategy json_llm --provider openai_compatible --model gpt-5.3-codex --base-url https://codex-api.packycode.com/v1 --wire-api responses
```

### Extra provider headers

```bash
python -m loop_agent.cli --goal "answer in json protocol" --strategy json_llm --provider openai_compatible --model gpt-5.3-codex --base-url https://codex-api.packycode.com/v1 --wire-api responses --provider-header "x-tenant:my-team" --provider-header "x-trace-id:demo-1"
```

### Retries and fallback model

```bash
python -m loop_agent.cli --goal "answer in json protocol" --strategy json_llm --provider openai_compatible --model gpt-5.3-codex --fallback-model gpt-5-codex --base-url https://codex-api.packycode.com/v1 --wire-api responses --max-retries 3 --retry-backoff-s 1.0 --retry-http-code 502 --retry-http-code 503
```

### Anthropic

```bash
set ANTHROPIC_API_KEY=sk-ant-xxx
python -m loop_agent.cli --goal "answer in json protocol" --strategy json_llm --provider anthropic --model claude-3-opus-20240229
```

### Gemini

```bash
set GEMINI_API_KEY=xxx
python -m loop_agent.cli --goal "answer in json protocol" --strategy json_llm --provider gemini --model gemini-pro
```

## Run Recording and Memory

LoopAgent records run data by default.

- Run records: `runs/<timestamp>/`
- Memory root: `.loopagent/runs/<run_id>/`
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

Load specific skills:

```bash
python -m loop_agent.agent_cli code --goal "search for info" --skill web_search --skill memory
```

## Project Layout

- `agents/`: guided learning-path scripts from minimal loop to team/subagent orchestration
- `skills/`: built-in skill notes and extension references
- `src/loop_agent/`: core package
- `src/loop_agent/core/`: generic loop engine, stop rules, and serialization
- `src/loop_agent/steps/`: reusable step strategies and registries
- `src/loop_agent/memory/`: JSONL memory store and context loading
- `src/loop_agent/llm/`: provider adapters
- `src/loop_agent/ops/`: provider doctor, git, and GitHub operational helpers
- `src/loop_agent/tool_use_loop.py`: the central model -> tools -> results loop
- `src/loop_agent/ui/`: optional chat-oriented TUI entrypoints
- `tests/`: unit tests
- `examples/`: optional demos and integrations
- `bin/loopagent.js`: npm bridge entrypoint
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
- `npm i -g @t0ugh-sys/loopagent` works only after the package has been published to npm.
