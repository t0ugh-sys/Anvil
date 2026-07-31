# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `anvil/ui/tool_renderer.py` — structured tool-call boxes rendered after each tool execution (`[r] read_file`, `[>] run_command`, etc.) using `chrome.py`'s `box_lines()`; wired into session runtime via `--tool-render` / `--no-tool-render` flag
- `docs/architecture.md` — data flow diagram (LLM → Loop → Tools → Compress), module table, layered config resolution, multi-agent overview
- 5-minute quick-start section in `README.md` referencing runnable examples
- `examples/hello_agent/`, `examples/multi_agent_team/`, `examples/cost_aware_agent/`, `examples/batch_processing/` — four runnable examples with READMEs

## [0.2.0] — 2026-07-28

### Added
- **Anthropic streaming** — `anthropic_stream_invoke_factory` with `on_chunk` callback; Rich Chat Anthropic provider streams tokens live with "Working..." → separator → live output transition
- **Async LLM provider** — `anthropic_async_invoke_factory()` / `_anthropic_async_invoke_factory()` using `asyncio.to_thread`; `AsyncInvokeFn` type in `_types.py`; zero new dependencies
- **Batch API client** — `AnthropicBatchClient` + `BatchRequest` in `anvil/llm/anthropic/batch.py`; 50% cost savings for bulk tasks; polling and result aggregation
- **Dynamic pricing** — `pricing.json` local cache; `/pricing` slash command to view or override per-model pricing; fallback to built-in static table on load failure
- **Integration tests** — `tests/integration/` with `test_full_loop.py`, `test_compression_e2e.py`, `test_batch_workflow.py`, `test_team_coordination.py`
- **Windows CI matrix** — `ubuntu-latest` + `windows-latest` × Python 3.10/3.12/3.13 in `.github/workflows/tests.yml`
- **Config schema validation** — `anvil/config/schema.py`; `validate_or_exit()` on startup; `build_layered_config(..., validate=True)`
- **Unified exception hierarchy** — `AnvilError` base; `RateLimitError(retry_after=...)`, `AuthError`, `ModelNotFoundError`, `ProviderTimeoutError`, `ProviderResponseError`; HTTP→semantic mapping in `_http.py`
- **Rate limit awareness** — `RateLimitTracker` reads `anthropic-ratelimit-*` response headers; shared across the session; visible in `/status`
- **Session/run GC** — `/gc [--dry-run] [--keep-days N] [--keep-count N]` slash command
- **Loop observability** — `LoopEvent` TypedDict (`tool_call`, `tool_result`, `compaction`, `cost_update`, `error`) emitted per round; `render_loop_stats()` shown in `/status`
- **Prompt cache** — `cache_control: {"type": "ephemeral"}` injected on eligible messages; cache hit/miss stats in `/status`
- **Token usage tracking** — `TokenUsageTracker` accumulates `input_tokens` / `output_tokens` / `cache_read_tokens`; `render_cache_summary()` in session renderer
- **TUI Rich Chat** — `RichLog` replaces `Static` widget (proper scrolling + Markdown); LLM calls offloaded to `asyncio.to_thread` (no UI freeze); Markdown rendering in `_print_response`; token progress bar in status bar
- **Parallel tool execution** — `ThreadPoolExecutor` with up to 8 workers in `_dispatch_tool_calls`
- **CircuitBreaker, PII filtering, importance scoring** — Zero2Agent safety and quality improvements
- **Extended thinking** — `thinking: {"type": "enabled", "budget_tokens": N}` pass-through for Anthropic provider

### Changed
- `providers.py` (73 KB monolith) split into `anvil/llm/anthropic/` package: `client.py`, `stream.py`, `batch.py`, `cache.py`, `tokens.py`, `_helpers.py`
- Tool count reduced from 32 → 12 built-in tools
- `CompressionConfig` renamed to `CompactConfig`; old name still accepted via alias
- Model list updated: `claude-opus-5`, `claude-sonnet-5`, `claude-sonnet-4-5`, `claude-haiku-4-5-20251001`

### Fixed
- Windows GBK encoding errors for Unicode characters in output (replaced emoji/special chars with ASCII-safe alternatives)
- `asyncio` event loop policy on Windows (`WindowsProactorEventLoopPolicy`)
- `HybridTokenCounter` calibration bug

## [0.1.0] — initial release

- Core tool-use loop (`anvil/agent/loop.py`)
- Multi-provider LLM support: Anthropic, Gemini, OpenAI-compatible, mock
- Context compression (four strategies)
- Multi-agent team runtime (`PersistentTeamRuntime`, `TaskGraph`, mailbox)
- Permission system (`PermissionManager`, `ToolPolicy`, `SecurityMonitor`)
- 12 built-in tools: files, shell, search, memory, git, GitHub CLI, todo, browser
- Rich Chat interactive UI and Textual TUI
- Session recording to `.anvil/runs/<run_id>/`
- Skill system (`skills/` directory)
