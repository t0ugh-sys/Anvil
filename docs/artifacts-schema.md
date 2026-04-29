# Run Artifacts Schema

This document describes the on-disk artifacts produced by Anvil runs.

Goals:
- Make run outputs stable to parse (CI, dashboards, replay tools).
- Allow forward evolution via explicit `schema_version`.

## Locations

By default, runs are written under:

- `.anvil/runs/<run_id>/`

Where `<run_id>` is a UTC timestamp like `20260306T010203Z`.

## Common Conventions

- All JSON files are UTF-8.
- All timestamps are UTC ISO-8601 strings.
- Producers must write `schema_version` so readers can branch by version.

Current schema:

- `schema_version`: `run-schema-v1`

Compatibility guidance:
- Patch changes (non-breaking): add new optional fields; keep existing field meanings.
- Breaking changes: bump `schema_version` and keep old readers working where practical.

## Artifact: events.jsonl

Path:
- `events.jsonl`

Format:
- Newline-delimited JSON (one object per line).
- Each line is an `EventRow` envelope.

### EventRow (run-schema-v1)

```json
{
  "schema_version": "run-schema-v1",
  "ts": "2026-03-06T01:02:03.456789+00:00",
  "event": "step_succeeded",
  "step": 3,
  "payload": {
    "step": 3,
    "output": "..."
  }
}
```

Fields:
- `schema_version` (string, required): schema id.
- `ts` (string, required): event timestamp (UTC ISO-8601).
- `event` (string, required): event name.
- `step` (integer or null, required): extracted from `payload.step` when present.
- `payload` (object, required): event-specific body.

Notes:
- The envelope is stable; `payload` may evolve per event type.
- Parsers should ignore unknown fields.

## Artifact: state.json

Path:
- `state.json`

Purpose:
- A small mutable snapshot used as working memory.

### State (run-schema-v1)

```json
{
  "schema_version": "run-schema-v1",
  "goal": "...",
  "step_index": 3,
  "last_output": "...",
  "history_tail": ["...", "..."]
}
```

Fields:
- `schema_version` (string, required)
- `goal` (string, required)
- `step_index` (integer, required)
- `last_output` (string, required)
- `history_tail` (array of strings, required): last N step outputs.

Notes:
- If an older `state.json` is missing `schema_version`, the loader will inject the current `SCHEMA_VERSION`.

## Artifact: summary.json

Path:
- `summary.json`

Purpose:
- A derived summary for providing long-term context across steps.

### Summary (run-schema-v1)

```json
{
  "schema_version": "run-schema-v1",
  "goal": "...",
  "current_plan": ["..."],
  "facts": ["..."],
  "work_done": ["..."],
  "open_questions": ["..."],
  "next_actions": ["..."],
  "steps": 3
}
```

Fields:
- `schema_version` (string, required)
- `goal` (string, required)
- `current_plan` (array of strings, required)
- `facts` (array of strings, required)
- `work_done` (array of strings, required)
- `open_questions` (array of strings, required)
- `next_actions` (array of strings, required)
- `steps` (integer, required)

Notes:
- The summarizer trims `work_done` and `open_questions` to the last 20 items.
- `next_actions` defaults to the first 3 items of `current_plan`.
