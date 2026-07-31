# Built-in Tools

Anvil ships 12 built-in tools executed inside the workspace sandbox.
All tools run through `execute_tool_call()` in `anvil/tools/__init__.py`, which
enforces permission checking and runs calls in parallel via a thread pool.

## Tool Reference

### `read_file`
Read a file from the workspace.

```json
{"name": "read_file", "arguments": {"path": "relative/path"}}
```

Returns the file content as a string.  Path must be inside the workspace root.

### `write_file`
Write or overwrite a file.  Creates parent directories automatically.

```json
{"name": "write_file", "arguments": {"path": "relative/path", "content": "text"}}
```

Use `content: ""` to create an empty file.

### `apply_patch`
Apply a unified diff patch to an existing file.

```json
{
  "name": "apply_patch",
  "arguments": {
    "patch": "*** Begin Patch\n*** Update File: path\n...\n*** End Patch"
  }
}
```

### `search`
Literal-text search across all workspace files.

```json
{"name": "search", "arguments": {"pattern": "literal text"}}
```

Returns matching file paths and line numbers.

### `run_command`
Run a subprocess synchronously (shell=False).

```json
{"name": "run_command", "arguments": {"cmd": ["git", "status"]}}
```

`cmd` must be a list — never a shell string.  Returns combined stdout/stderr.

### `run_command_async`
Start a subprocess in the background and return immediately.

```json
{"name": "run_command_async", "arguments": {"cmd": ["python", "server.py"]}}
```

### `todo_write`
Persist the agent's structured todo list in session state.

```json
{
  "name": "todo_write",
  "arguments": {
    "items": [{"id": "1", "content": "Fix bug", "status": "pending"}]
  }
}
```

### `compact`
Trigger context compression mid-run to free up context window space.

```json
{"name": "compact", "arguments": {}}
```

### `load_skill`
Load a named skill definition from `skills/` into the context.

```json
{"name": "load_skill", "arguments": {"name": "skill-name"}}
```

### `web_search`
Search the web via DuckDuckGo (requires the `browser` extra).

```json
{"name": "web_search", "arguments": {"query": "search terms"}}
```

### `fetch_url`
Fetch and return the text content of a URL.

```json
{"name": "fetch_url", "arguments": {"url": "https://example.com"}}
```

### `analyze_memory`
Inspect JSONL memory entries from past runs.

```json
{"name": "analyze_memory", "arguments": {"query": "optional filter text"}}
```

## Permissions

Each tool call is checked against the session's `ToolPolicy` before execution.
Three built-in policies are available:

| Policy | Effect |
|---|---|
| `ToolPolicy.allow_all()` | All tools allowed |
| `ToolPolicy.read_only()` | Only read/search tools |
| `ToolPolicy.deny_all()` | No tools |

Custom rules can be added via `ToolPolicy.add_rule(name, allow=True/False)`.

The interactive runtime also has a `permission_mode` (ask / allow / deny) that
gates tool calls through user prompts at runtime.

## Adding Custom Tools

Register a `ToolSpec` with `register_tool(spec, handler)` before creating the
agent.  A `ToolSpec` requires a `name`, `description`, and JSON Schema `parameters`.
The handler receives `(arguments: dict) -> str`.

## Git / GitHub Tools

GitHub CLI wrappers (`gh` subcommands) are available in `anvil/ops/github/` but
are not registered as built-in tools by default.  Import and register the ones
you need, or invoke them via `run_command` with `["gh", ...]`.
