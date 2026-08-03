# Anvil 优化规划 · 2026-07-28

> **项目定位**：Terminal-first coding agent runtime，灵感来自 Claude Code。  
> **当前版本**：0.1.0 Alpha · Python 3.10+ / Node.js 18+ 桥接  
> **本文目标**：系统性梳理优化方向，按优先级排列，给出具体可行的实施路径。

---

## 目录

1. [现状评估](#1-现状评估)
2. [P0 — 阻断性问题](#2-p0--阻断性问题)
3. [P1 — 核心质量提升](#3-p1--核心质量提升)
4. [P2 — 性能与成本优化](#4-p2--性能与成本优化)
5. [P3 — 开发者体验](#5-p3--开发者体验)
6. [P4 — 长期架构演进](#6-p4--长期架构演进)
7. [实施路线图](#7-实施路线图)
8. [成功指标](#8-成功指标)

---

## 1. 现状评估

### 优势

| 维度 | 状态 |
|------|------|
| 工具调用循环 | ✅ 成熟，`tool_use_loop.py` 29KB，逻辑集中 |
| LLM 适配层 | ✅ 多 Provider（Anthropic / Gemini / OpenAI-compat / Mock） |
| 上下文压缩 | ✅ 四种压缩策略，`compression.py` 49KB |
| 成本控制 | ✅ `CostTracker` + Batch API（节省 50%） |
| 多 Agent 协作 | ✅ `team_runtime.py`、`scheduler.py`、`mailbox.py`、`worktree_manager.py` |
| 测试覆盖 | ✅ 45+ 个单元测试文件 |
| 权限治理 | ✅ `PermissionManager` + `ToolPolicy` + `SecurityMonitor` |

### 主要问题

| 编号 | 问题 | 影响 |
|------|------|------|
| G1 | `providers.py` 单文件 73KB，职责过重 | 可维护性差，PR 冲突频繁 |
| G2 | 运行时全同步，`ThreadPoolExecutor` 模拟并发 | 多 Agent 场景延迟高 |
| G3 | 无集成/端到端测试 | 回归风险大 |
| G4 | `CostTracker` 定价硬编码 | 价格变动后静默错误 |
| G5 | `examples/` 只有 README，无可运行示例 | 新用户上手困难 |
| G6 | `CHANGELOG.md` 停在 v0.1.0（2025-03-05） | 版本追踪缺失 |
| G7 | Batch API 无持久化任务队列/Webhook 通知 | 长任务可靠性不足 |
| G8 | 无 async/await 全链路支持 | 流式响应阻塞主线程 |
| G9 | Windows CI 缺失 | asyncio 在 Windows 行为差异未被测试覆盖 |
| G10 | 异常类型混用（裸 `Exception`/`ValueError`/`RuntimeError`） | 错误无法分级处理，日志噪声大 |
| G11 | API 速率限制无感知 | 多 Agent 并发时大量无效重试 |
| G12 | `.anvil/sessions/` + `.anvil/runs/` 无 GC | 长期使用磁盘无限增长 |
| G13 | `layered_config.py` 无 schema 校验 | 配置错误静默失效或运行时 KeyError |
| G14 | `rich_chat.py` 响应用纯文本渲染，Markdown 已导入未使用；token 进度条参数未传入，永远空 | 界面信息密度低，成本可视化失效 |
| G15 | `tui_chat.py` 用 `Static` 累积消息字符串，`async` 方法内同步阻塞 LLM 调用 | UI 随使用内存增长，响应期间整个 TUI 冻结 |

---

## 2. P0 — 阻断性问题

> 必须在下一个公开版本前修复，否则影响基本可用性或安全性。

### 2.1 拆分 `providers.py`（73KB → 模块包）✅ 已实现

**现状**：所有 Provider 逻辑堆在一个文件，包含 `CostTracker`、`BatchClient`、`TokenCounter`、`PromptCache`、流式工厂等。

**方案**：

```
src/anvil/llm/
├── __init__.py          # 公开 API 不变，向后兼容
├── base.py              # 抽象基类 + 公共类型
├── anthropic/
│   ├── __init__.py
│   ├── client.py        # 同步 invoke + chat
│   ├── stream.py        # SSE 流式工厂
│   ├── batch.py         # AnthropicBatchClient
│   ├── cache.py         # PromptCache + cache_control 注入
│   └── cost.py          # CostTracker + 定价表（见 2.2）
├── gemini.py
├── openai_compat.py
├── mock.py
└── retry.py             # 已独立，保持不动
```

**收益**：单文件 < 300 行，PR 冲突消失，各子模块独立测试。

---

### 2.2 `CostTracker` 定价动态化 ✅ 已实现

**现状**：`_resolve_pricing()` 中价格硬编码，Anthropic 调价后静默计算错误。

**方案**：

```python
# src/anvil/llm/anthropic/cost.py

PRICING_FILE = Path(__file__).parent / "pricing.json"   # 本地缓存
PRICING_URL  = "https://api.anthropic.com/v1/models"    # 官方端点（若开放）

class CostTracker:
    def _resolve_pricing(self, model: str) -> ModelPricing:
        # 1. 尝试从 pricing.json 读取（含 updated_at 字段）
        # 2. 若 > 7 天未更新，后台异步拉取最新定价
        # 3. 拉取失败则 fallback 到内置静态表 + 日志告警
        ...
```

定价更新已通过交互式 `/pricing` slash 命令实现，可查看或覆写 `pricing.json` 中的条目（`/pricing <model> <input> <output> <cw> <cr>`）。

---

### 2.3 补充关键集成测试 ✅ 已实现

**现状**：45+ 单元测试，零集成测试。

**最小化方案**（不依赖真实 API，用 `MockProvider`）：

```
tests/integration/
├── test_full_loop.py          # 完整工具调用循环（Mock LLM）
├── test_compression_e2e.py    # 超长上下文触发压缩后结果正确
├── test_batch_workflow.py     # Batch 提交→轮询→结果聚合
└── test_team_coordination.py  # 两个子 Agent 协作完成任务
```

目标：集成测试覆盖率 ≥ 4 个核心流程，CI 必须通过才能合并。

---

### 2.4 Windows CI 矩阵 ✅ 已实现

**现状**：`.github/workflows/tests.yml` 仅跑 Linux runner。项目的主要开发和使用环境是 Windows 11，而 `asyncio` 在 Windows 上默认使用 `ProactorEventLoop`，行为与 Linux `SelectorEventLoop` 存在差异（特别是子进程、管道、信号处理）。

**方案**：

```yaml
# .github/workflows/tests.yml
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest]
    python-version: ["3.10", "3.12"]

# Windows 专项检查项：
# - asyncio.get_event_loop_policy() 差异
# - background.py BackgroundCommandRunner（子进程）
# - worktree_manager.py（git 路径分隔符）
# - tool_use_loop.py ThreadPoolExecutor shutdown 行为
```

同时在 `src/anvil/` 启动时加入平台检测：

```python
import sys, asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
```

---

## 3. P1 — 核心质量提升

### 3.1 异步化关键路径 · 阶段一 ✅ 已实现

**现状**：全同步 + `ThreadPoolExecutor`。流式响应 `anthropic_stream_invoke_factory` 用同步 SSE 读取，阻塞调用线程。

**渐进式方案**（不做全量重写）：

1. **阶段一** ✅：`anvil/llm/_http.py` 新增 `_http_post_json_async()`（`asyncio.to_thread` 包裹同步 urllib，零新依赖）；`anvil/llm/anthropic/client.py` 新增 `anthropic_async_invoke_factory()` / `_anthropic_async_invoke_factory()`，对外暴露 `async def invoke(prompt: str) -> str`（`AsyncInvokeFn` 类型见 `_types.py`），内含独立的异步重试/退避逻辑；同步接口 `anthropic_invoke_factory()` 保持不变、未受影响。测试见 `tests/test_anthropic_async.py`。
2. **阶段二** ✅：`anvil/agent/loop.py` 新增 `_dispatch_tool_calls_async()`（`asyncio.gather()` + `asyncio.to_thread()` 替代 `ThreadPoolExecutor`），以及 `execute_tool_use_round_async` / `make_tool_use_step_async`；`anvil/coding_agent.py` 新增 `run_coding_agent_async()`；`session_runtime.py` 通过 `asyncio.run()` 接入。同步路径完全保留。测试 562/562 通过。
3. **阶段三** ✅：`anvil/runtime/team.py` 新增 `spawn_teammate_async`（`asyncio.create_task` 替代 `threading.Thread`）、`_run_teammate_loop_async`（`asyncio.sleep` + `await run_coding_agent_async`）、`shutdown_all_async`、`dispatch_ready_tasks_async`（`asyncio.Lock`）；同步 API 完全保留。测试见 `tests/test_team_runtime_async.py`（5 个），567/567 通过。

**风险**：Python 3.10 async 兼容性需验证，特别是 Windows 上的 `asyncio` 事件循环策略。阶段一使用 `asyncio.to_thread`，在 Windows `ProactorEventLoop` 下行为等同线程池，风险较低。

---

### 3.2 Batch API 可靠性增强 ✅ 已实现

**现状**：`anvil/llm/anthropic/batch.py` 中 `BatchJobStore` 已实现（SQLite 持久化，零额外依赖），`AnthropicBatchClient` 接受可选 `store: BatchJobStore | None` 参数：

```python
class BatchJobStore:
    """SQLite-backed 持久化，零额外依赖（stdlib sqlite3）"""
    def save(self, batch_id: str, model: str, metadata: dict) -> None: ...
    def load_pending(self) -> list[BatchJob]: ...
    def mark_done(self, batch_id: str, status: str = 'ended') -> None: ...

class AnthropicBatchClient:
    def __init__(self, ..., store: BatchJobStore | None = None): ...
    def submit(self, requests, metadata=None) -> str: ...       # 存在 store 时自动 save()
    def get_results(self, batch_id) -> list[BatchResult]: ...   # 完成后自动 mark_done()
    def cancel(self, batch_id) -> dict: ...                     # 取消后自动 mark_done(status='cancelled')
    def resume_pending_jobs(self) -> list[BatchJob]: ...        # 进程启动时调用，委托给 store.load_pending()
```

每个 SQLite 连接用完即关闭（避免 Windows 下文件句柄泄漏）。测试见 `tests/test_batch_job_store.py` 与 `tests/integration/test_batch_workflow.py::TestBatchClientWithStore`。

---

### 3.3 工具调用循环可观测性 ✅ 已实现

**现状**：`ToolUseState` 内部状态不透明，调试困难。

**方案**：引入结构化事件发射，不改变核心逻辑：

```python
class LoopEvent(TypedDict):
    type: Literal["tool_call", "tool_result", "compaction", "cost_update", "error"]
    timestamp: float
    data: dict

# tool_use_loop.py
self._emit(LoopEvent(type="tool_call", ...))  # 已有 hooks 机制扩展即可
```

配合 `events.jsonl` 落盘（现已有），在 `/status` 命令中展示循环统计（调用次数、耗时、成本分布）。

---

### 3.4 统一异常体系 ✅ 已实现

**实现**：`ProviderError(AnvilError, ValueError)` MRO 继承（向后兼容），`_map_http_error()` 在 `anvil/llm/_http.py` 实现 HTTP→语义异常映射（429→`RateLimitError(retry_after=...)`，401/403→`AuthError`，404→`ModelNotFoundError`，`URLError`→`ProviderTimeoutError`，其他→`ProviderResponseError`）。`Retry-After` 响应头在 `_http_post_json` 层解析并传入 `ProviderHttpError.retry_after`。`anthropic/client.py` 和 `gemini.py` 已移除裸 `ValueError` 转换。14 个新测试通过：`tests/test_provider_exceptions.py`。

---

### 3.5 API 速率限制感知 ✅ 已实现

**现状**：`RateLimitTracker`（`anvil/llm/rate_limit.py`）已实现并接入交互式运行时。从 Anthropic 响应头读取剩余请求数/token 数及重置时间，通过 `/status` 命令可查看当前速率限制状态。`InteractiveRuntime` 和 `build_interactive_turn_runner` 均已接受 `rate_limit_tracker` 参数并在整个会话中共享同一实例。

对多 Agent 场景（`team_runtime.py`），后续可将同一 `RateLimitTracker` 实例共享给所有子 Agent，协调整体请求速率（待 P1 异步化阶段完成后统一处理）。

---

### 3.6 配置 Schema 验证 ✅ 已实现

**现状**：`anvil/config/schema.py` 已实现完整的配置校验体系。`validate_config()` 检查所有已知字段的类型和取值范围，`validate_or_exit()` 在启动时 fast-fail，`build_layered_config(..., validate=True)` 可选开启校验。测试覆盖：`tests/test_config_schema.py`（12 个测试）。

**方案**（纯 stdlib，不引入 pydantic）：

```python
# src/anvil/config_schema.py

@dataclass
class AnvilConfig:
    model: str = "claude-opus-5"
    max_tokens: int = 8192
    provider: Literal["anthropic", "gemini", "openai_compat", "mock"] = "anthropic"
    permissions: PermissionConfig = field(default_factory=PermissionConfig)
    # ...

    def validate(self) -> list[str]:
        """返回错误列表，空列表表示有效"""
        errors = []
        if self.max_tokens < 1 or self.max_tokens > 200_000:
            errors.append(f"max_tokens {self.max_tokens} out of range [1, 200000]")
        # ...
        return errors
```

启动时调用 `validate()`，有错误则打印清单并 `sys.exit(1)`，而不是运行到一半崩溃。

---

### 3.7 会话与运行目录 GC ✅ 已实现

**现状**：`anvil/commands/slash.py` 中 `_execute_gc_command` 已实现 `/gc [--dry-run] [--keep-days 30] [--keep-count 100]`，支持按天数清理旧 sessions、按数量保留最近 runs、dry-run 预览。测试覆盖：`tests/test_slash_gc.py`（6 个测试）。

---

### 3.8 更新 CHANGELOG ✅ 已实现

按 [Keep a Changelog](https://keepachangelog.com/) 格式，补录自 v0.1.0 以来所有功能：

- `[Unreleased]` → Batch API, CostTracker, Streaming, Token Counting, Prompt Cache, Zero2Agent 系列改进（CircuitBreaker, PII filtering, importance scoring, parallel tools）

计划在下次正式发布时标记为 `v0.2.0`。

---

### 3.9 Rich Chat 三处低成本 Bug 修复 ✅ 已实现

**问题一：Markdown 渲染未启用**

[rich_chat.py:351-366](src/anvil/ui/rich_chat.py#L351-L366) 中 `_print_response()` 用的是纯文本 `response_lines()`，而 `rich.Markdown` 虽已导入却没用上。

```python
# 修复前
def _print_response(console, text, cfg):
    for line in response_lines(text, width=width):
        console.print(line, style='anvil.output', markup=False)

# 修复后
def _print_response(console, text, cfg):
    if HAS_RICH:
        console.print(Markdown(text))   # 代码块、加粗、列表全部正确渲染
    else:
        for line in response_lines(text, width=width):
            console.print(line, markup=False)
```

**问题二：token 进度条永远空**

[rich_chat.py:364](src/anvil/ui/rich_chat.py#L364) 调用 `status_bar()` 时缺少 `tokens_used`/`max_tokens`，需将这两个值从 LLM 响应中提取并传入。

```python
# status_bar 调用补全 token 信息
usage = getattr(reply_obj, 'usage', None)
sb = status_bar(
    cfg.model, provider_label, str(Path.cwd()),
    width=width,
    tokens_used=getattr(usage, 'input_tokens', 0),
    max_tokens=cfg.max_tokens,
)
```

**问题三：模型列表过时**

[rich_chat.py:139](src/anvil/ui/rich_chat.py#L139) Anthropic 候选列表停在 `claude-3-5-sonnet-latest`，补充当前可用模型：

```python
if provider == 'anthropic':
    return [
        'claude-opus-5',
        'claude-sonnet-5',
        'claude-sonnet-4-5',
        'claude-haiku-4-5-20251001',
        'claude-3-5-sonnet-latest',
    ]
```

---

### 3.10 TUI 两处根本性问题修复 ✅ 已实现

**问题一：`Static` → `RichLog`（消息累积）**

[tui_chat.py:427](src/anvil/ui/tui_chat.py#L427) 当前用单个 `Static` widget 拼接所有消息（`existing + reply`），消息越多内存越大，且无真正滚动。

```python
# 修复：换用 RichLog（Textual 内置，支持行级滚动 + Markdown）
from textual.widgets import RichLog

# compose() 中：
yield RichLog(id='log', markup=True, highlight=True, auto_scroll=True)

# 追加消息时：
log = self.query_one('#log', RichLog)
log.write(Markdown(reply))          # 支持代码高亮
log.write(f'[bold green]> {text}[/bold green]')
```

**问题二：同步阻塞 → `asyncio.to_thread()`**

[tui_chat.py:574](src/anvil/ui/tui_chat.py#L574) `async def on_input_submitted` 内直接调同步 `current_invoke()`，阻塞 Textual 事件循环导致 UI 冻结。

```python
# 修复：卸载到线程池
async def on_input_submitted(self, event: Input.Submitted) -> None:
    text = event.value.strip()
    event.input.value = ''
    if not text:
        return

    log = self.query_one('#log', RichLog)
    log.write(f'[bold green]> {text}[/bold green]')
    log.write('[dim]Working...[/dim]')

    import asyncio
    try:
        messages = load_messages(current_cfg.history_limit)
        reply = await asyncio.to_thread(current_invoke, messages)
    except Exception as e:
        reply = f'ERROR: {e}'

    log.write(Markdown(reply))
```

---

## 4. P2 — 性能与成本优化

### 4.1 Prompt Cache 命中率优化 ✅ 已实现

**现状**：`PromptCacheManager` 已实现 stable prefix + dynamic suffix 拆分，但 cache-control 注入时机依赖手动调用。

**优化**：
- 在 `tool_use_loop.py` 中自动检测消息稳定段，注入 `cache_control: {"type": "ephemeral"}`
- 添加命中率监控：`CacheStats(hits, misses, saved_tokens, saved_cost)`
- `/status` 命令展示本次会话的 cache 节省统计

**预期收益**：长会话中 prompt tokens 成本降低 30–60%。

---

### 4.2 工具集精简与路由 ✅ 已实现

**现状**：Zero2Agent 已将工具从 32 个精简到 12 个。进一步优化：

- 实现 **lazy tool loading**：仅在用户请求相关能力时将工具定义注入 prompt（当前全量注入）
- 工具描述自动截断到 200 tokens（现部分描述过长）
- 为高频工具（`read_file`, `write_file`, `shell`）添加调用结果缓存（TTL=30s，幂等操作）

---

### 4.3 上下文压缩策略调优 ✅ 已实现

**现状**：`CompactManager` 有四种策略，但触发阈值是静态配置。

**优化**：
- 引入动态阈值：根据当前模型的 context window 大小（从 Provider 元数据读取）自动调整
- `micro-compact` 优先压缩工具调用结果（通常占 60%+ token），保留对话轮次
- 添加压缩质量评估：压缩后用 LLM 验证关键信息是否保留（仅在 `--quality-check` 模式下，避免增加成本）

---

### 4.4 Token 估算精度提升 ✅ 已实现

**现状**：`HybridTokenCounter` 本地估算 + API 计数 fallback。

**优化**：
- 缓存模型 tokenizer 配置（`tiktoken`-style BPE 表），提升本地估算精度到 ±3%
- 在 `--dry-run` 模式下输出预估 token 数和成本，方便用户决策

---

### 4.5 性能基准套件 ✅ 已实现

**现状**：无基准测试，任何重构都无法量化性能影响，回归无从发现。

**方案**（零外部依赖，stdlib `timeit` + `statistics`）：

```
tests/benchmarks/
├── bench_tool_loop.py        # 工具调用循环吞吐（ops/sec）
├── bench_compression.py      # 各压缩策略耗时 vs 压缩率
├── bench_token_estimation.py # HybridTokenCounter 精度 vs 速度
└── bench_results.json        # CI 写入，PR 时对比（阈值：±10% 触发告警）
```

CI 集成：

```yaml
# .github/workflows/tests.yml
- name: Run benchmarks
  run: python -m tests.benchmarks.run_all --output bench_results.json
- name: Compare with baseline
  run: python scripts/compare_benchmarks.py bench_results.json baseline.json
```

**目标**：工具调用循环基线 > 50 ops/sec（Mock LLM），压缩策略耗时 < 200ms/10K tokens。

---

### 4.6 Rich Chat 流式输出 ✅ 已实现

**现状**：等待完整响应后一次性渲染，对长回复体验差（无感知进度）。

**方案**：利用已有的 `anthropic_stream_invoke_factory`，在 Rich Chat 中接入逐 token 打印：

```python
# rich_chat.py — 替换同步 invoke 为流式版本
def _print_streaming_response(console: Console, stream_iter, cfg: ChatConfig) -> str:
    """逐 chunk 打印，返回完整文本"""
    full_text = []
    width = _ui_width(console)
    console.print(f'  {separator_line(width - 4)}', style='anvil.separator')
    console.print(f'  {RESPONSE_MARKER} ', style='anvil.response', end='')

    for chunk in stream_iter:          # anthropic_stream_invoke_factory 的 yield
        console.print(chunk, style='anvil.output', end='', markup=False)
        full_text.append(chunk)

    console.print()                    # 换行
    console.print(f'  {separator_line(width - 4)}', style='anvil.separator')
    return ''.join(full_text)
```

**降级策略**：Provider 不支持流式时（Gemini、OpenAI-compat 部分模型）自动 fallback 到原有同步调用，对用户透明。

---

## 5. P3 — 开发者体验

### 5.1 可运行示例 ✅ 已实现

在 `examples/` 下添加：

```
examples/
├── README.md                    # 现有
├── hello_agent/
│   ├── run.py                   # 最小化 agent：读文件、回答问题
│   └── README.md
├── batch_processing/
│   ├── run.py                   # 批量代码审查（Batch API）
│   └── README.md
├── multi_agent_team/
│   ├── run.py                   # 两个 Agent 协作重构代码
│   └── README.md
└── cost_aware_agent/
    ├── run.py                   # 带预算限制的 agent
    └── README.md
```

---

### 5.2 依赖安装体验 ✅ 已实现

**现状**：`dependencies = []`，用户需手动安装 `anthropic` 等 SDK（实际上 `providers.py` 用 raw urllib，不需要 SDK）。

**优化**：

```toml
# pyproject.toml
[project.optional-dependencies]
anthropic = ["anthropic>=0.40.0"]
gemini    = ["google-generativeai>=0.8.0"]
openai    = ["openai>=1.50.0"]
tui       = ["textual>=0.86.2", "rich>=13.0"]
browser   = ["playwright>=1.40.0"]
all       = ["anvil[anthropic,gemini,openai,tui,browser]"]
dev       = ["anvil[all]", "pytest>=8.0", "pytest-asyncio>=0.24"]
```

对应文档更新：`pip install anvil[anthropic]` 即可使用 Anthropic provider。

---

### 5.3 `anvil doctor`（已移除）

`ops/doctor.py` 和对应的 CLI 子命令已在"Delete CLI"重构中一并删除。健康检查功能后续可通过 `/status` slash 命令扩展，或在首次启动时以警告形式内联输出（无需独立命令）。

---

### 5.4 文档补全 ✅ 已实现

| 文档 | 现状 | 行动 |
|------|------|------|
| `docs/architecture.md` | `repo-layout`, `learning-path` 存在 | 补充数据流图（LLM → Loop → Tools → Compress） |
| `docs/providers.md` | 缺失 | 新增：各 Provider 配置方法、环境变量列表 |
| `docs/tools.md` | 缺失 | 新增：内置工具列表、自定义工具开发指南 |
| `docs/team.md` | 缺失 | 新增：多 Agent 团队使用指南 |
| `CHANGELOG.md` | 停在 v0.1.0 | 见 3.4 |
| `README.md` | 存在 | 添加快速开始（5 分钟内运行第一个 agent） |

---

### 5.5 主 Agent 工具调用结构化渲染 ✅ 已实现

**现状**：`tool_use_loop.py` 执行工具时全靠裸 `print`，用户在终端看不到工具名称、参数摘要、耗时、成功/失败。体验远不如 Claude Code。

**方案**：在现有 hooks 机制上加一层渲染层（不改 loop 核心逻辑）：

```
┌── 🔧 read_file ─────────────────────────────────────┐
│  path: src/anvil/tool_use_loop.py                   │
│  ✓ 完成  29KB · 0.12s                               │
└─────────────────────────────────────────────────────┘

┌── 💻 shell ──────────────────────────────────────────┐
│  python -m pytest tests/ -x -q                      │
│  ✓ 完成  47 passed · 2.3s                            │
└─────────────────────────────────────────────────────┘

┌── ✏️  write_file ────────────────────────────────────┐
│  path: src/anvil/exceptions.py  (+142 lines)        │
│  ✓ 完成  0.04s                                       │
└─────────────────────────────────────────────────────┘
```

实现位于 `PostToolUse` hook，使用 `chrome.py` 已有的 `box_lines()` + 颜色系统，零新依赖：

```python
# src/anvil/ui/tool_renderer.py
from .chrome import box_lines, CHECK_MARK, CROSS_MARK, colorize, ...

TOOL_ICONS = {
    'read_file': '📄', 'write_file': '✏️ ', 'shell': '💻',
    'search': '🔍', 'memory': '🧠', 'web_search': '🌐',
}

def render_tool_call(tool_name: str, args: dict, result, elapsed_s: float, *, color: bool) -> list[str]:
    icon = TOOL_ICONS.get(tool_name, '🔧')
    title = f'{icon} {tool_name}'
    lines = [_summarize_args(tool_name, args)]
    status = f'{CHECK_MARK} 完成  {elapsed_s:.2f}s'
    lines.append(colorize(status, GREEN, enabled=color))
    return box_lines(lines, width=80, title=title)
```

**开关**：通过 `--tool-render` 标志或 `layered_config.py` 配置 `ui.tool_render: true`，默认开启，`--quiet` 时关闭。

---

## 6. P4 — 长期架构演进

### 6.1 Plugin 系统 ✅ 已实现

将 `skills/` 中的技能合约演进为正式插件系统：

```python
# 第三方包可以注册技能
# setup.cfg / pyproject.toml:
# [project.entry-points."anvil.skills"]
# my_skill = "my_package.skill:MySkill"

class SkillBase(Protocol):
    name: str
    description: str
    def get_tools(self) -> list[ToolDef]: ...
    def on_activate(self, ctx: SessionContext) -> None: ...
```

`discover_plugins()` 使用 `importlib.metadata.entry_points(group='anvil.skills')` 发现已安装的第三方技能，`SkillLoader._load_external()` 优先尝试 entry points，再 fallback 到 `anvil_skills.*` 命名空间包。16 个测试见 `tests/test_plugin_system.py`。commit `220671a`。

---

### 6.2 协议层 (`protocols/`) 标准化 ✅ 已实现

`anvil/protocols/mcp.py` 实现 MCP (Model Context Protocol) 兼容层：

- 内容类型：`MCPTextContent`、`MCPImageContent`
- 工具描述：`MCPTool`（含 JSON Schema `inputSchema`）、`MCPInputSchema`
- 调用/结果：`MCPToolCallParams`、`MCPCallToolResult`
- 双向转换：`anvil_tool_def_to_mcp`、`mcp_result_from_anvil`、`parse_mcp_tool_call`、`tool_list_to_mcp`

第三方 MCP 客户端可直接列举并调用 Anvil 工具，无需了解内部格式。29 个测试见 `tests/test_mcp_protocol.py`。commit `3651f90`。

---

### 6.3 可观测性平台集成 ✅ 已实现

`anvil/observability/` 模块提供：

- **tracing.py**：零硬依赖 OTel 包装器，OTel 未安装时自动降级为 `NoOpTracer`；`@trace_tool_call` / `@trace_llm_invoke` 装饰器支持同步和异步函数，记录 `tool.name`、`tool.ok`、`elapsed_ms` 等 span 属性
- **logging.py**：纯 stdlib 结构化日志，`JSONFormatter` 输出 NDJSON，`StructuredLogger` 支持关键字参数字段，`configure_json_logging()` 幂等安装

33 个测试见 `tests/test_observability.py`。commit `faf026d`。

---

### 6.4 Web UI（可选）

当前有 TUI（Textual）和 Rich Chat。长期可考虑：
- FastAPI + WebSocket 后端，将 `chat_runtime` 暴露为 HTTP API
- 轻量 Web UI（htmx 或 React），与现有 TUI 并列，不替代

---

### 6.5 语义记忆（Memory 模块升级）

**现状**：`memory/` 使用 JSONL 平铺存储，检索依赖全量扫描 + 关键词匹配，对长期积累的上下文利用率低。

**方案**：引入向量索引，零外部服务依赖（本地文件）：

```python
# src/anvil/memory/vector_store.py

class VectorMemoryStore:
    """
    sqlite-vec（stdlib sqlite3 + 向量扩展）或 纯 Python 近似 KNN
    - 写入时：将记忆内容 embed（本地 embedding 或 Anthropic embed API）
    - 检索时：query embed → cosine similarity top-k
    - 落盘：.anvil/memory/vectors.db（SQLite）
    """
    def add(self, key: str, text: str, metadata: dict): ...
    def search(self, query: str, top_k: int = 5) -> list[MemoryResult]: ...
    def delete(self, key: str): ...
```

分级策略：
- **短期记忆**：当前会话内，存 `ToolUseState`，不持久化
- **长期记忆**：跨会话，存 `VectorMemoryStore`，启动时加载 top-5 相关记忆注入 system prompt
- **项目记忆**：与 git repo 绑定，`.anvil/memory/` 可提交到版本控制共享给团队

**预期收益**：跨会话的知识复用，减少重复解释项目背景，长期任务续接更顺畅。

---

## 7. 实施路线图

```
2026 Q3（当前季度）
├── Week 1-2  │ [P0] 拆分 providers.py → llm/ 子包
├── Week 2-3  │ [P0] CostTracker 动态定价 + pricing.json
├── Week 3-4  │ [P0] 集成测试框架 + 4 个核心流程
├── Week 3-4  │ [P0] Windows CI 矩阵（ubuntu + windows × py3.10/3.12）
└── Week 4    │ [P1] 统一异常体系（AnvilError 层级） + CHANGELOG v0.2.0 发布

2026 Q4
├── Month 1   │ [P1] 异步化阶段一（providers async + asyncio Windows 策略）✅
├── Month 1   │ [P1] API 速率限制感知（RateLimitTracker + 429 精确等待）
├── Month 1   │ [P1] Rich Chat bug 修复（Markdown渲染、token进度条、模型列表）✅
├── Month 1   │ [P1] TUI 重构（RichLog替换Static + asyncio.to_thread）✅
├── Month 1   │ [P2] Prompt Cache 命中率监控
├── Month 2   │ [P1] Batch API 持久化（SQLite BatchJobStore）✅
├── Month 2   │ [P1] 配置 Schema 验证（启动时 fast-fail）
├── Month 2   │ [P2] Rich Chat 流式输出（逐 token 打字效果）
├── Month 2   │ [P3] examples/ 四个示例
├── Month 3   │ [P1] 工具循环可观测性 + /status 增强
├── Month 3   │ [P1] /gc slash 命令（会话/运行目录清理）
├── Month 3   │ [P2] 性能基准套件 + CI 对比
├── Month 3   │ [P3] 主 agent 工具调用结构化渲染（tool_renderer.py）
└── Month 3   │ [P3] 文档补全（4 篇新文档）

2027 Q1
├── Month 1   │ [P1] 异步化阶段二（tool_use_loop async）
├── Month 2   │ [P4] Plugin 系统 v1
├── Month 2   │ [P4] protocols/ 梳理 + MCP 对齐
└── Month 3   │ [P4] OpenTelemetry 集成 + 语义记忆 v1（VectorMemoryStore）
```

---

## 8. 成功指标

| 指标 | 当前值 | 目标值 | 截止 |
|------|--------|--------|------|
| 最大单文件行数 | 73KB (providers.py) | < 500 行 | Q3 |
| 集成测试数量 | 0 | ≥ 4 流程 | Q3 |
| CI 平台矩阵 | Linux only | Linux + Windows | Q3 |
| Prompt Cache 节省率（长会话） | 未监控 | ≥ 30% token | Q4 |
| `anvil doctor` 检查项 | 已移除（ops/doctor.py 随 CLI 一并删除） | — | — |
| 文档覆盖（核心模块） | ~40% | ≥ 80% | Q4 |
| 可运行示例数量 | 0 | ≥ 4 | Q4 |
| CHANGELOG 更新频率 | 停滞 17 个月 | 每 release 更新 | Q3 起 |
| 429 重试等待精度 | 盲目指数退避 | 读取 retry-after header | Q4 |
| 会话目录 GC | 无 | `/gc` slash 命令可用 + 自动策略 | Q4 |
| 基准测试工具循环吞吐 | 未测量 | > 50 ops/sec（Mock） | Q4 |
| Rich Chat Markdown 渲染 | 未启用 | 正确渲染代码块/加粗/列表 | Q4 |
| TUI LLM 调用阻塞 | 冻结 UI | asyncio.to_thread 非阻塞 | Q4 |xxx
| 语义记忆检索支持 | 无 | VectorMemoryStore v1 | 2027 Q1 |

---

*生成时间：2026-07-28 · 基于 `main` 分支（704a855）*
