---
name: memory
description: Analyze past runs and learn patterns
---

# memory

Use `memory` when the agent must preserve durable task state instead of relying
only on the last few conversational turns.

Persistent artifacts:

- `events.jsonl`
- `state.json`
- `summary.json`

Rules:

- Keep long-term facts in summaries.
- Keep short-term details in recent step windows.
- Rebuild context from persisted state before long runs or resumed runs.
