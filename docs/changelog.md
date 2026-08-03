# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Multi-agent orchestration**: dependency-aware task graph and scheduler, subagent runtime, persistent mailbox, isolated git worktrees per subagent, capability-based tool policies
- **Claude API optimizations**: prompt caching (`PromptCache`), extended thinking passback, streaming responses, native token counting, stop sequences, Batch API client with ~50% cost savings, dynamic `CostTracker` pricing backed by `llm/pricing.json`
- **Context compression**: multi-layer compaction engine (micro/partial/hierarchical strategies) with persisted checkpoints
- **Interactive runtime**: Claude Code-style terminal session with slash commands (`/help`, `/status`, `/history`, `/todo`, `/tools`, `/model`, `/pricing`, `/exit`), permission modes, session persistence and resume
- **GitHub tools**: `gh` CLI wrappers for repo/issue/PR workflows
- **Reliability**: circuit breaker for repeated tool failures, PII filtering, security monitor, input sanitization, importance-scored context retention
- **Rich terminal UI**: boxed tool-call rendering, status bar with token usage, themed chrome components; optional Textual-based TUI with runtime provider/model pickers
- Windows CI matrix (Python 3.10–3.13 on ubuntu-latest and windows-latest)
- Provider-specific optional dependency extras in `pyproject.toml` (`browser`, `yaml`, `tui`, `chat`, `all`, `dev`)
- Config schema validation (`anvil/config/schema.py`): `validate_config()`/`validate_or_exit()` catch unknown fields and out-of-range values in layered config before they turn into runtime `KeyError`s; opt in via `build_layered_config(..., validate=True)`
- Batch API persistence: `BatchJobStore` (SQLite-backed, stdlib `sqlite3`) survives process restarts; `AnthropicBatchClient` accepts an optional `store` to auto-save on `submit()`, mark done on `get_results()`/`cancel()`, and `resume_pending_jobs()` on startup
- Session/run GC: `/gc [--dry-run] [--keep-days N] [--keep-count N]` slash command cleans up old `.anvil/sessions/` and `.anvil/runs/` directories; dry-run mode shows what would be removed without deleting; current session is always preserved
- Async Anthropic invoke: `anthropic_async_invoke_factory()` / `_anthropic_async_invoke_factory()` in `anvil/llm/anthropic/client.py` expose an `AsyncInvokeFn`, backed by `_http_post_json_async()` (`asyncio.to_thread` wrapping the existing sync urllib call); sync `invoke()` is unchanged
- Unified provider exception hierarchy: `ProviderError(AnvilError, ValueError)` MRO makes all provider errors catchable as `ValueError` for backward compatibility; `_map_http_error()` in `anvil/llm/_http.py` maps HTTP status codes to semantic types (`RateLimitError` with `retry_after`, `AuthError`, `ModelNotFoundError`, `ProviderTimeoutError`, `ProviderResponseError`); `Retry-After` response header parsed and propagated; `urllib.error.URLError` now raises `ProviderTimeoutError` instead of propagating uncaught

### Changed

- Reorganized package layout: `src/anvil/` → `anvil/` at repo root; internal subpackages `infra/`, `config/`, `agent/`, `runtime/`, `compression/`; `docker/`, `docs/`, `examples/` split out of the repo root
- Split the single `providers.py` module into an `llm/` subpackage with per-provider modules (`anthropic/`, `gemini.py`, `openai_compat.py`, `mock.py`)
- Split `anvil/ops/github_tools.py` into a `github/` subpackage (`_shared.py`, `repo.py`, `issues.py`, `pulls.py`); `github_tools.py` is now a backward-compatibility shim
- Reduced built-in tool count from 32 to 12; tool execution now runs in parallel via a thread pool
- Removed backward-compatibility shims after the package reorg; all internal imports use canonical subpackage paths

### Fixed

- Rich Chat (`anvil-chat`) status bar now shows real token usage against the provider's context window instead of an empty progress bar

### Removed

- Batch/CLI interface (`anvil/cli.py`, `anvil/agent_cli.py`, `anvil doctor`, `update-pricing` subcommand) in favor of the interactive runtime; pricing lookups now live in the `/pricing` slash command

## [0.1.0] - 2025-03-05

### Added

- **LLM Providers**
  - Anthropic (Claude) support via `anthropic` provider
  - Google Gemini support via `gemini` provider
  - Provider registry with `list_providers()` and `get_provider()` functions

- **Skill System**
  - Pluggable skill architecture (`src/anvil/skills.py`)
  - Built-in skills: `web_search`, `memory`, `files`, `commands`, `browser`
  - Support for custom third-party skills
  - Dynamic skill loading via `SkillLoader`

- **Tools**
  - `web_search` - Search the web using DuckDuckGo
  - `fetch_url` - Fetch and parse web page content
  - `analyze_memory` - Analyze past runs for learning patterns
  - Browser automation tools (navigate, click, fill, screenshot, evaluate)
  - Safe command execution with `cmd` list parameter (shell=False)

- **Configuration System** (`src/anvil/config.py`)
  - YAML configuration file support
  - JSON configuration file support
  - `.env` file support for API keys
  - Config merging and validation

- **Logging System** (`src/anvil/logging.py`)
  - Structured logging with multiple levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  - Multiple output destinations (stdout, stderr, file, jsonl)
  - Logger context helpers (`log_step`, `log_tool`, `log_event`)

- **Prompt Templates** (`src/anvil/prompts.py`)
  - Reusable prompt templates with variable substitution
  - Built-in templates: `json_loop`, `coding`, `analyze`, `research`
  - Custom template registration
  - Template loading from YAML/JSON files

- **Error Handling** (`src/anvil/errors.py`)
  - Comprehensive error hierarchy (`AnvilError` base class)
  - Specialized exceptions: `ConfigError`, `ProviderError`, `ToolError`, `ValidationError`, `MemoryError`, `SkillError`
  - Input validation functions
  - Error formatting for JSON output

- **High-level API** (`src/anvil/api.py`)
  - `AgentConfig` dataclass with validation
  - `AgentResult` for run results
  - `AnvilAPI` class for easy integration
  - `create_agent()` and `run_goal()` convenience functions

- **Docker Support**
  - `Dockerfile` for containerized deployment
  - `docker-compose.yml` for local development

### Changed

- **Python Version**: Support Python 3.10+ (was 3.11+)
- **CI**: Updated to test Python 3.10, 3.11, 3.12
- **Core Package**: Remains stdlib-only (no external dependencies)

### Fixed

- Browser skill import path in `skills.py`
- Registered `BrowserSkill` for user access

### Security

- Added safe command execution mode using `cmd` (list of arguments) instead of `command` (string with shell=True)

## [0.0.1] - 2024-01-01

### Added

- Initial release
- Core Anvil engine
- Basic LLM providers (mock, openai_compatible)
- File tools (read_file, write_file, apply_patch, search)
- CLI interface
- JSON loop strategy
- Memory system

[Unreleased]: https://github.com/t0ugh-sys/Anvil/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/t0ugh-sys/Anvil/releases/tag/v0.1.0
[0.0.1]: https://github.com/t0ugh-sys/Anvil/releases/tag/v0.0.1
