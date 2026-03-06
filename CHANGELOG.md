# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-03-05

### Added

- **LLM Providers**
  - Anthropic (Claude) support via `anthropic` provider
  - Google Gemini support via `gemini` provider
  - Provider registry with `list_providers()` and `get_provider()` functions

- **Skill System**
  - Pluggable skill architecture (`src/loop_agent/skills.py`)
  - Built-in skills: `web_search`, `memory`, `files`, `commands`, `browser`
  - Support for custom third-party skills
  - Dynamic skill loading via `SkillLoader`

- **Tools**
  - `web_search` - Search the web using DuckDuckGo
  - `fetch_url` - Fetch and parse web page content
  - `analyze_memory` - Analyze past runs for learning patterns
  - Browser automation tools (navigate, click, fill, screenshot, evaluate)
  - Safe command execution with `cmd` list parameter (shell=False)

- **Configuration System** (`src/loop_agent/config.py`)
  - YAML configuration file support
  - JSON configuration file support
  - `.env` file support for API keys
  - Config merging and validation

- **Logging System** (`src/loop_agent/logging.py`)
  - Structured logging with multiple levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  - Multiple output destinations (stdout, stderr, file, jsonl)
  - Logger context helpers (`log_step`, `log_tool`, `log_event`)

- **Prompt Templates** (`src/loop_agent/prompts.py`)
  - Reusable prompt templates with variable substitution
  - Built-in templates: `json_loop`, `coding`, `analyze`, `research`
  - Custom template registration
  - Template loading from YAML/JSON files

- **Error Handling** (`src/loop_agent/errors.py`)
  - Comprehensive error hierarchy (`LoopAgentError` base class)
  - Specialized exceptions: `ConfigError`, `ProviderError`, `ToolError`, `ValidationError`, `MemoryError`, `SkillError`
  - Input validation functions
  - Error formatting for JSON output

- **High-level API** (`src/loop_agent/api.py`)
  - `AgentConfig` dataclass with validation
  - `AgentResult` for run results
  - `LoopAgentAPI` class for easy integration
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
- Core LoopAgent engine
- Basic LLM providers (mock, openai_compatible)
- File tools (read_file, write_file, apply_patch, search)
- CLI interface
- JSON loop strategy
- Memory system

[Unreleased]: https://github.com/t0ugh-sys/LoopAgent/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/t0ugh-sys/LoopAgent/releases/tag/v0.1.0
[0.0.1]: https://github.com/t0ugh-sys/LoopAgent/releases/tag/v0.0.1
