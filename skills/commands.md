# commands

Use `commands` when the agent needs local verification such as tests, linters,
or project-specific scripts.

Provided tools:

- `run_command`

Rules:

- Prefer structured `cmd` arrays over shell strings.
- Treat command execution as observable state; capture outputs in run traces.
- Keep command scope inside the active workspace.
