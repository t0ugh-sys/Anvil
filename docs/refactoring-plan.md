# Anvil 全项目重构计划

> 基于 2026-07-28 代码审查，17,443 行 Python，54 个测试文件，35+ 平铺模块

---

## 现状问题

### 顶层目录混乱
- Python项目 / Node包装器 / Docker / 文档 / 示例全堆在根目录
- `Dockerfile` 和 `docker-compose.yml` 散落根目录
- 重要文档 (`OPTIMIZATION_PLAN.md`, `CONTRIBUTING.md`, `CHANGELOG.md`) 与代码混放

### Python包结构过于扁平
- `src/anvil/` 根目录有 35+ 个松散模块，职责边界不清
- 超大文件：`compression.py` (1460行), `tool_use_loop.py` (780行), `team_runtime.py` (626行)
- 职责重叠：`session.py` vs `services/session_runtime.py`，`cli.py` vs `agent_cli.py` vs `entrypoints/agent.py`

### 测试结构不匹配源码
- `tests/` 54个测试文件全在一级目录，无子目录分组
- 测试路径不镜像源码结构，难以定位和维护

### 文档分散
- `docs/` 只有3个文件
- 关键文档散落根目录

---

## 目标结构

```
Anvil/
├── .github/
│   └── workflows/
│       ├── tests.yml         # 更新测试路径
│       └── release.yml
├── anvil/                    # 主Python包 (从 src/anvil/ 移动)
│   ├── __init__.py
│   ├── core/                 # 核心agent runtime (保持)
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── types.py
│   │   └── serialization.py
│   ├── llm/                  # LLM提供商抽象 (保持)
│   │   ├── __init__.py
│   │   ├── anthropic/
│   │   ├── gemini.py
│   │   ├── openai_compat.py
│   │   ├── cache.py
│   │   ├── mock.py
│   │   ├── usage.py
│   │   ├── pricing.json
│   │   ├── _cli.py
│   │   ├── _http.py
│   │   ├── _types.py
│   │   └── providers.py      # re-export shim，最终删除
│   ├── agent/                # 新包：Agent执行引擎
│   │   ├── __init__.py
│   │   ├── loop.py           # 从 tool_use_loop.py 拆分 — 主循环
│   │   ├── executor.py       # 从 tool_use_loop.py 拆分 — 工具执行
│   │   ├── protocol.py       # 从 agent_protocol.py
│   │   ├── subagents.py      # 从 subagents.py
│   │   └── background.py     # 从 background.py
│   ├── runtime/              # 新包：运行时服务
│   │   ├── __init__.py
│   │   ├── chat.py           # 从 services/chat_runtime.py
│   │   ├── coding.py         # 从 services/coding_runtime.py
│   │   ├── session.py        # 合并 session.py + services/session_runtime.py
│   │   ├── team.py           # 从 team_runtime.py 拆分
│   │   ├── mailbox.py        # 从 mailbox.py
│   │   ├── scheduler.py      # 从 scheduler.py
│   │   ├── task_graph.py     # 从 task_graph.py
│   │   ├── task_store.py     # 从 task_store.py
│   │   └── run_recorder.py   # 从 run_recorder.py
│   ├── config/               # 新包：配置系统
│   │   ├── __init__.py
│   │   ├── loader.py         # 从 config.py
│   │   ├── layered.py        # 从 layered_config.py
│   │   └── schema.py         # 合并 context_schema.py + run_schema.py
│   ├── infra/                # 新包：基础设施横切关注点
│   │   ├── __init__.py
│   │   ├── hooks.py          # 从 hooks.py
│   │   ├── skills.py         # 从 skills.py
│   │   ├── permissions.py    # 从 permissions.py
│   │   ├── policies.py       # 从 policies.py
│   │   ├── retry.py          # 从 retry.py
│   │   └── logging.py        # 从 logging.py
│   ├── compression/          # 新包：从 compression.py (1460行) 拆分
│   │   ├── __init__.py
│   │   ├── micro.py          # micro_compact_messages
│   │   ├── partial.py        # partial_compact_messages
│   │   ├── hierarchical.py   # 层级压缩策略
│   │   └── manager.py        # CompressionManager 统一入口
│   ├── tools/                # 工具实现 (保持结构)
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── file_tools.py
│   │   ├── search_tools.py
│   │   ├── command_tools.py
│   │   └── memory_tools.py
│   ├── ops/                  # Git/GitHub 操作 (拆分大文件)
│   │   ├── __init__.py
│   │   ├── git_tools.py
│   │   ├── github/           # 从 github_tools.py (602行) 拆分
│   │   │   ├── __init__.py
│   │   │   ├── issues.py
│   │   │   ├── pulls.py
│   │   │   └── repo.py
│   │   ├── worktree.py       # 从 worktree_manager.py
│   │   └── doctor.py
│   ├── memory/               # 持久化存储 (保持)
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── jsonl_store.py
│   ├── ui/                   # 用户界面 (业务逻辑剥离)
│   │   ├── __init__.py
│   │   ├── rich_chat.py      # 精简，只保留UI代码
│   │   ├── tui_chat.py       # 修复已知TUI bug
│   │   └── chrome.py
│   ├── commands/             # Slash命令 (保持)
│   │   ├── __init__.py
│   │   └── slash.py
│   ├── steps/                # Step策略 (保持)
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── json_loop.py
│   │   └── demo.py
│   ├── entrypoints/          # CLI入口 (保持，删除重复)
│   │   ├── __init__.py
│   │   ├── agent.py          # 主入口 anvil
│   │   └── parser_builders.py
│   ├── errors.py             # 异常体系 (保持，全局统一引用)
│   ├── messages.py           # 消息类型
│   ├── prompts.py            # Prompt模板
│   ├── token_estimation.py   # Token计数
│   ├── todo.py               # TODO追踪
│   ├── utils.py              # 公共工具函数
│   └── api.py                # API端点
├── tests/
│   ├── _bootstrap.py         # 保持
│   ├── conftest.py           # 新：pytest fixtures
│   ├── unit/                 # 新：单元测试，镜像源码结构
│   │   ├── core/
│   │   │   ├── test_agent.py
│   │   │   └── test_serialization.py
│   │   ├── llm/
│   │   │   ├── test_provider_anthropic_gemini.py
│   │   │   ├── test_provider_fallback.py
│   │   │   ├── test_provider_headers.py
│   │   │   ├── test_claude_api_features.py
│   │   │   └── test_claude_api_optimizations.py
│   │   ├── agent/
│   │   │   ├── test_tool_use_loop.py
│   │   │   ├── test_agent_protocol.py
│   │   │   ├── test_subagents.py
│   │   │   └── test_background.py
│   │   ├── runtime/
│   │   │   ├── test_session.py
│   │   │   ├── test_session_enhanced.py
│   │   │   ├── test_chat_runtime.py
│   │   │   ├── test_team_runtime.py
│   │   │   ├── test_task_graph.py
│   │   │   ├── test_task_store.py
│   │   │   ├── test_scheduler.py
│   │   │   └── test_mailbox.py
│   │   ├── config/
│   │   │   ├── test_config.py
│   │   │   ├── test_layered_config.py
│   │   │   └── test_context_schema.py
│   │   ├── infra/
│   │   │   ├── test_hooks.py
│   │   │   ├── test_skills.py
│   │   │   ├── test_permissions.py
│   │   │   ├── test_policies.py
│   │   │   └── test_retry.py
│   │   ├── compression/
│   │   │   └── test_compression.py
│   │   ├── tools/
│   │   │   ├── test_tools.py
│   │   │   ├── test_git_tools.py
│   │   │   └── test_github_tools.py
│   │   ├── memory/
│   │   │   └── test_memory_store.py
│   │   ├── ui/
│   │   │   └── test_tui_chat.py
│   │   └── cli/
│   │       ├── test_cli.py
│   │       ├── test_agent_cli.py
│   │       └── test_api.py
│   └── integration/          # 保持，增加更多
│       ├── __init__.py
│       ├── test_full_loop.py
│       ├── test_compression_e2e.py
│       ├── test_batch_workflow.py
│       └── test_team_coordination.py
├── docs/                     # 整合所有文档
│   ├── index.md              # 项目概览
│   ├── architecture.md       # 新：架构设计文档
│   ├── refactoring-plan.md   # 本文档
│   ├── optimization-plan.md  # 从 OPTIMIZATION_PLAN.md
│   ├── contributing.md       # 从 CONTRIBUTING.md
│   ├── changelog.md          # 从 CHANGELOG.md
│   ├── artifacts-schema.md   # 已有
│   ├── learning-path.md      # 已有
│   └── repo-layout.md        # 已有，更新内容
├── examples/
│   ├── README.md
│   ├── basic/
│   │   └── json_loop_stub_demo.py
│   ├── llm/
│   │   ├── provider_demo.py
│   │   ├── logging_demo.py
│   │   └── prompts_demo.py
│   └── advanced/
│       ├── api_demo.py
│       └── browser_tools.py
├── skills/                   # 保持
│   ├── README.md
│   ├── browser/
│   ├── commands/
│   ├── files/
│   ├── memory/
│   └── web_search/
├── docker/                   # 新：Docker相关独立目录
│   ├── Dockerfile            # 从根目录移动
│   └── docker-compose.yml    # 从根目录移动
├── scripts/
│   └── py310_compat.py
├── bin/                      # Node包装器 (保持)
│   └── anvil.js
├── pyproject.toml            # 更新包路径 (src/anvil → anvil)
├── package.json
├── README.md                 # 精简，指向 docs/
├── LICENSE
├── .gitignore                # 更新
├── .env.example
└── requirements.txt
```

---

## 文件迁移映射表

### 顶层结构变动

| 旧路径 | 新路径 | 操作 |
|--------|--------|------|
| `src/anvil/` | `anvil/` | 移动，更新pyproject.toml |
| `Dockerfile` | `docker/Dockerfile` | 移动 |
| `docker-compose.yml` | `docker/docker-compose.yml` | 移动 |
| `OPTIMIZATION_PLAN.md` | `docs/optimization-plan.md` | 移动 |
| `CONTRIBUTING.md` | `docs/contributing.md` | 移动 |
| `CHANGELOG.md` | `docs/changelog.md` | 移动 |

### anvil/ 内部迁移

| 旧路径 | 新路径 | 操作 |
|--------|--------|------|
| `tool_use_loop.py` | `agent/loop.py` + `agent/executor.py` | 拆分 |
| `agent_protocol.py` | `agent/protocol.py` | 移动 |
| `subagents.py` | `agent/subagents.py` | 移动 |
| `background.py` | `agent/background.py` | 移动 |
| `team_runtime.py` | `runtime/team.py` | 移动 |
| `session.py` | `runtime/session.py` | 合并session_runtime.py |
| `services/chat_runtime.py` | `runtime/chat.py` | 移动 |
| `services/coding_runtime.py` | `runtime/coding.py` | 移动 |
| `services/session_runtime.py` | 合并到 `runtime/session.py` | 合并 |
| `mailbox.py` | `runtime/mailbox.py` | 移动 |
| `scheduler.py` | `runtime/scheduler.py` | 移动 |
| `task_graph.py` | `runtime/task_graph.py` | 移动 |
| `task_store.py` | `runtime/task_store.py` | 移动 |
| `run_recorder.py` | `runtime/run_recorder.py` | 移动 |
| `config.py` | `config/loader.py` | 移动 |
| `layered_config.py` | `config/layered.py` | 移动 |
| `context_schema.py` | `config/schema.py` | 合并run_schema.py |
| `run_schema.py` | 合并到 `config/schema.py` | 合并 |
| `hooks.py` | `infra/hooks.py` | 移动 |
| `skills.py` | `infra/skills.py` | 移动 |
| `permissions.py` | `infra/permissions.py` | 移动 |
| `policies.py` | `infra/policies.py` | 移动 |
| `retry.py` | `infra/retry.py` | 移动 |
| `logging.py` | `infra/logging.py` | 移动 |
| `compression.py` | `compression/` (拆分为4个文件) | 拆分 |
| `ops/github_tools.py` | `ops/github/` (拆分为3个文件) | 拆分 |
| `worktree_manager.py` | `ops/worktree.py` | 移动 |
| `agent_cli.py` | 已删除（与 `cli.py` 一同移除） | 删除 |
| `runtime.py` | 分散到 `runtime/` 各模块 | 拆分分散 |
| `coding_agent.py` | `runtime/coding.py` | 合并 |

### 删除的文件（内容合并后）

| 文件 | 原因 |
|------|------|
| `src/anvil/agent_cli.py` | 已删除（与 `cli.py` 一同移除） |
| `src/anvil/run_schema.py` | 合并到 `anvil/config/schema.py` |
| `src/anvil/coding_agent.py` | 合并到 `anvil/runtime/coding.py` |
| `src/anvil/services/session_runtime.py` | 合并到 `anvil/runtime/session.py` |
| `src/anvil/services/` 目录 | 内容分散到 `anvil/runtime/` |

### tests/ 迁移（54个文件 → tests/unit/ + tests/integration/）

| 旧路径 | 新路径 |
|--------|--------|
| `tests/test_agent.py` | `tests/unit/core/test_agent.py` |
| `tests/test_agent_cli.py` | `tests/unit/cli/test_agent_cli.py` |
| `tests/test_agent_protocol.py` | `tests/unit/agent/test_agent_protocol.py` |
| `tests/test_api.py` | `tests/unit/cli/test_api.py` |
| `tests/test_background.py` | `tests/unit/agent/test_background.py` |
| `tests/test_chat_runtime.py` | `tests/unit/runtime/test_chat_runtime.py` |
| `tests/test_claude_api_features.py` | `tests/unit/llm/test_claude_api_features.py` |
| `tests/test_claude_api_optimizations.py` | `tests/unit/llm/test_claude_api_optimizations.py` |
| `tests/test_cli.py` | `tests/unit/cli/test_cli.py` |
| `tests/test_coding_agent.py` | `tests/unit/runtime/test_coding_agent.py` |
| `tests/test_compression.py` | `tests/unit/compression/test_compression.py` |
| `tests/test_config.py` | `tests/unit/config/test_config.py` |
| `tests/test_context_schema.py` | `tests/unit/config/test_context_schema.py` |
| `tests/test_doctor.py` | `tests/unit/tools/test_doctor.py` |
| `tests/test_git_tools.py` | `tests/unit/tools/test_git_tools.py` |
| `tests/test_github_tools.py` | `tests/unit/tools/test_github_tools.py` |
| `tests/test_hooks.py` | `tests/unit/infra/test_hooks.py` |
| `tests/test_json_loop_step.py` | `tests/unit/agent/test_json_loop_step.py` |
| `tests/test_json_protocol.py` | `tests/unit/agent/test_json_protocol.py` |
| `tests/test_layered_config.py` | `tests/unit/config/test_layered_config.py` |
| `tests/test_learning_path.py` | `tests/unit/core/test_learning_path.py` |
| `tests/test_mailbox.py` | `tests/unit/runtime/test_mailbox.py` |
| `tests/test_memory_store.py` | `tests/unit/memory/test_memory_store.py` |
| `tests/test_permissions.py` | `tests/unit/infra/test_permissions.py` |
| `tests/test_policies.py` | `tests/unit/infra/test_policies.py` |
| `tests/test_provider_anthropic_gemini.py` | `tests/unit/llm/test_provider_anthropic_gemini.py` |
| `tests/test_provider_fallback.py` | `tests/unit/llm/test_provider_fallback.py` |
| `tests/test_provider_headers.py` | `tests/unit/llm/test_provider_headers.py` |
| `tests/test_retry.py` | `tests/unit/infra/test_retry.py` |
| `tests/test_scheduler.py` | `tests/unit/runtime/test_scheduler.py` |
| `tests/test_serialization.py` | `tests/unit/core/test_serialization.py` |
| `tests/test_session.py` | `tests/unit/runtime/test_session.py` |
| `tests/test_session_enhanced.py` | `tests/unit/runtime/test_session_enhanced.py` |
| `tests/test_skills.py` | `tests/unit/infra/test_skills.py` |
| `tests/test_step_registry.py` | `tests/unit/agent/test_step_registry.py` |
| `tests/test_subagents.py` | `tests/unit/agent/test_subagents.py` |
| `tests/test_task_graph.py` | `tests/unit/runtime/test_task_graph.py` |
| `tests/test_task_store.py` | `tests/unit/runtime/test_task_store.py` |
| `tests/test_team_runtime.py` | `tests/unit/runtime/test_team_runtime.py` |
| `tests/test_todo.py` | `tests/unit/infra/test_todo.py` |
| `tests/test_token_estimation.py` | `tests/unit/core/test_token_estimation.py` |
| `tests/test_tool_use_loop.py` | `tests/unit/agent/test_tool_use_loop.py` |
| `tests/test_tools.py` | `tests/unit/tools/test_tools.py` |
| `tests/test_tui_chat.py` | `tests/unit/ui/test_tui_chat.py` |
| `tests/test_worktree_manager.py` | `tests/unit/tools/test_worktree_manager.py` |
| `tests/test_zero2agent_improvements.py` | `tests/unit/core/test_zero2agent_improvements.py` |
| `tests/test_zero2agent_round3.py` | `tests/unit/core/test_zero2agent_round3.py` |
| `tests/test_zero2agent_round4.py` | `tests/unit/core/test_zero2agent_round4.py` |

---

## 执行计划

### Phase 0 — 准备工作 (预计 2小时)

- [ ] 当前分支跑完整测试，记录 baseline 数量 (当前 319 个)
- [ ] 创建新分支 `refactor/project-structure`
- [ ] 确认 `.gitignore` 包含 `dist/`, `.anvil/`, `__pycache__/`

### Phase 1 — 顶层重组 (预计 半天)

- [ ] `src/anvil/` → `anvil/` (移动整个目录)
- [ ] 更新 `pyproject.toml`：`package-dir = {"" = "."}` (删除 `src` 层)
- [ ] 创建 `docker/` 目录，移动 `Dockerfile` 和 `docker-compose.yml`
- [ ] 整合文档到 `docs/`
- [ ] 整理 `examples/` 子目录
- [ ] 更新 CI：`.github/workflows/tests.yml` 测试路径
- [ ] 运行测试，确认 319 全通过

### Phase 2 — anvil/ 内部重组 (预计 1天)

- [ ] 创建子包目录：`agent/`, `runtime/`, `config/`, `infra/`, `compression/`
- [ ] 逐模块移动文件（按迁移映射表）
- [ ] 在旧位置添加 re-export shim，确保测试不破坏
- [ ] 拆分 `compression.py` → `compression/` 4个文件
- [ ] 拆分 `tool_use_loop.py` → `agent/loop.py` + `agent/executor.py`
- [ ] 拆分 `team_runtime.py` → `runtime/team.py`（清理）
- [ ] 拆分 `ops/github_tools.py` → `ops/github/` 3个文件
- [ ] 合并 `session.py` + `services/session_runtime.py`
- [ ] 删除 `cli.py` 和 `agent_cli.py`（已在"Delete CLI"重构中完成）
- [ ] 运行测试，确认全通过

### Phase 3 — tests/ 重组 (预计 半天)

- [ ] 创建 `tests/unit/` 子目录结构（镜像 `anvil/` 子包）
- [ ] 按映射表逐一移动测试文件
- [ ] 更新各测试文件中的 import 路径
- [ ] 删除 `tests/_bootstrap.py` 中的旧 sys.path hacks（如有）
- [ ] 运行测试，确认全通过

### Phase 4 — 清理 (预计 2小时)

- [ ] 删除所有 re-export shim（或保留以维持公开API向后兼容）
- [ ] 删除合并后的冗余文件
- [ ] 更新 `README.md`（精简，指向 docs/）
- [ ] 更新 `docs/repo-layout.md` 反映新结构
- [ ] 最终完整测试运行
- [ ] PR 合并到 main

---

## 风险与注意事项

| 风险 | 缓解措施 |
|------|----------|
| 大量 import 路径破坏 | 每个 Phase 后立即运行完整测试套件 |
| re-export shim 遗漏 | 使用 `grep -r "from anvil.X import"` 验证所有引用 |
| Windows路径问题 | Phase 1 后同步补 Windows CI 矩阵 |
| 测试文件移动后找不到 `_bootstrap.py` | 统一改用 `conftest.py` + pytest fixtures |
| 循环导入 | 每次移动后用 `python -c "import anvil"` 快速验证 |

---

## 成功指标

- [ ] `python -m pytest tests/` 全部通过（目标 320+ 个测试）
- [ ] `python -m pip install -e .` 成功，`anvil --help` 可运行
- [ ] `src/` 目录不再存在
- [ ] `anvil/` 根目录下直接模块 ≤ 10 个（当前 35+）
- [ ] `tests/` 根目录下直接测试文件 = 0（全在 `unit/` 或 `integration/`）
- [ ] Docker 相关全在 `docker/`
- [ ] 文档全在 `docs/`
