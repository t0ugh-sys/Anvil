# files

Use `files` when the agent must inspect or modify repository content safely
inside the configured workspace root.

Provided tools:

- `read_file`
- `write_file`
- `apply_patch`
- `search`

Rules:

- Paths must remain inside the workspace root.
- Prefer `apply_patch` for targeted edits.
- Prefer `search` before broad file traversal.
