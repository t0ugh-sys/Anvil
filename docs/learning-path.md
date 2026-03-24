# Learning Path

LoopAgent now exposes a staged path inspired by `learn-claude-code`, but adapted
to this repository's architecture.

## Stages

1. Minimal loop engine: `agents/s01_loop.py`
2. JSON protocol flow: `agents/s02_protocol.py`
3. Persistent memory: `agents/s03_memory.py`
4. Skill loading: `agents/s04_skills.py`
5. Coding-agent execution: `agents/s05_coding.py`
6. Team orchestration: `agents/s06_team.py`
7. Consolidated view: `agents/s_full.py`

## Why This Layout

- New users can understand the system incrementally.
- Production code stays under `src/loop_agent/`.
- Example scripts stay thin and reuse real runtime code.
- The repo reads like a harness project, not a one-off demo.
- The orchestration layer stays explicit instead of being hidden in prompt text.
