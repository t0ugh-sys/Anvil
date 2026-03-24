# LoopAgent Agents

This directory mirrors the staged learning style used by `learn-claude-code`,
but the examples are adapted to LoopAgent's runtime and APIs.

## Order

1. `s01_loop.py`: understand the raw loop engine
2. `s02_protocol.py`: understand structured JSON decisions
3. `s03_memory.py`: inspect persisted memory and summaries
4. `s04_skills.py`: inspect skill loading and capability boundaries
5. `s05_coding.py`: run a coding-agent style tool loop
6. `s06_team.py`: inspect task graph, mailbox, and sub-agent coordination
7. `s_full.py`: see the complete harness view in one script

## Run

```bash
python agents/s01_loop.py
python agents/s02_protocol.py
python agents/s03_memory.py
python agents/s04_skills.py
python agents/s05_coding.py
python agents/s06_team.py
python agents/s_full.py
```

All stages use mock or local components by default, so they can run without
external API keys.
