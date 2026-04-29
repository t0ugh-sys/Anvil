# Learning Path

Anvil keeps its staged concepts in the production runtime and docs, instead of
shipping separate walkthrough scripts.

## Stages

1. Minimal loop engine: `src/loop_agent/core/`
2. JSON protocol flow: `src/loop_agent/agent_protocol.py`
3. Persistent memory: `src/loop_agent/memory/`
4. Skill loading: `src/loop_agent/skills.py`
5. Coding-agent execution: `src/loop_agent/coding_agent.py`
6. Team orchestration: `src/loop_agent/subagents.py` and `src/loop_agent/scheduler.py`
7. Worktree isolation: `src/loop_agent/worktree_manager.py`
8. Consolidated runtime loop: `src/loop_agent/tool_use_loop.py`

## Why This Layout

- New users can still understand the system incrementally.
- Production code stays under `src/loop_agent/`.
- The repo avoids shipping disposable walkthrough scripts as first-class surface area.
- The orchestration layer stays explicit instead of being hidden in prompt text.
