# LoopAgent

LoopAgent is a controllable agent harness for iterative execution, tool calling,
memory persistence, and coding-agent style runs.

This repository now follows a layout closer to `learn-claude-code`: a small
runtime core, a progressive `agents/` learning path, a `skills/` capability
layer, and focused documentation.

## Python Requirement

- Requires Python `3.10+`
- Recommended: Python `3.11` or `3.12`

## Quick Start

```bash
python -m pip install -e .
python -m unittest discover -s tests -p "test_*.py" -v
python -m loop_agent.agent_cli tools
python -m loop_agent.agent_cli code --goal "inspect README then finish" --workspace . --provider mock --model mock-v3 --output json
```

## Install From GitHub Via npm

```bash
npm i -g git+https://github.com/t0ugh-sys/LoopAgent.git
loopagent tools
loopagent code --goal "inspect README then finish" --workspace . --provider mock --model mock-v3 --output json
```

`@t0ugh-sys/loopagent` will only work after publishing to the npm registry.

## Learn The Project

The repository is organized as a progressive learning path instead of a single
large demo:

- `agents/`: runnable stages from minimal loop to coding agent
- `skills/`: capability notes for built-in skills and extension rules
- `docs/`: repo layout, run artifacts, and learning path documentation
- `src/loop_agent/`: production runtime, CLI, tools, memory, providers
- `examples/`: API and integration examples outside the guided path

Start here:

1. Read [agents/README.md](D:\workspace\LoopAgent\agents\README.md)
2. Run [agents/s01_loop.py](D:\workspace\LoopAgent\agents\s01_loop.py)
3. Move through the rest of the `agents/` stages in order

## Guided Agent Stages

| Stage | Focus | File |
| --- | --- | --- |
| S01 | Minimal iterative loop | `agents/s01_loop.py` |
| S02 | Structured JSON protocol | `agents/s02_protocol.py` |
| S03 | Persistent memory and summaries | `agents/s03_memory.py` |
| S04 | Skills and capability loading | `agents/s04_skills.py` |
| S05 | Coding agent and tools | `agents/s05_coding.py` |
| S06 | Task graph, mailbox, sub-agents | `agents/s06_team.py` |
| FULL | Combined harness view | `agents/s_full.py` |

## Main CLI

General loop runner:

```bash
python -m loop_agent.cli --goal "write a one-line self introduction" --strategy demo --output json
```

Coding-agent runner:

```bash
python -m loop_agent.agent_cli code --goal "fix failing test" --workspace . --provider mock --model mock-v3 --output json
```

List built-in skills:

```bash
python -m loop_agent.agent_cli skills
```

Run with explicit skills:

```bash
python -m loop_agent.agent_cli code --goal "inspect the repository" --workspace . --provider mock --model mock-v3 --skill files --skill memory --output json
```

## What LoopAgent Already Supports

- Controlled loop execution with explicit stop reasons
- Structured coding-agent protocol with tool calls
- Workspace-safe tools such as `read_file`, `write_file`, `apply_patch`, `search`, and `run_command`
- Persistent run traces in `.loopagent/runs/<run_id>/`
- State summary injection with `state_summary` and `last_steps`
- Swappable providers for mock, OpenAI-compatible, Anthropic, and Gemini flows
- Built-in skills for files, commands, memory, web search, and optional browser automation
- First-pass orchestration primitives for task graphs, async mailbox coordination, and sub-agent dispatch

## Skills

Built-in skills are documented under [skills/README.md](D:\workspace\LoopAgent\skills\README.md).

Current built-ins:

- `files`
- `commands`
- `memory`
- `web_search`
- `browser` (optional dependency)

## Run Artifacts

Each recorded run can write:

- `events.jsonl`: event stream for replay/debugging
- `state.json`: latest state snapshot
- `summary.json`: compact long-term summary

Artifact details are documented in
[docs/artifacts-schema.md](D:\workspace\LoopAgent\docs\artifacts-schema.md).

## Repository Docs

- [docs/learning-path.md](D:\workspace\LoopAgent\docs\learning-path.md)
- [docs/repo-layout.md](D:\workspace\LoopAgent\docs\repo-layout.md)
- [examples/README.md](D:\workspace\LoopAgent\examples\README.md)

## Testing

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Conda example:

```powershell
conda --no-plugins run --no-capture-output -n base python -m unittest discover -s tests -p "test_*.py" -v
```

## License

MIT
