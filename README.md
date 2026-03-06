# LoopAgent

> Requires **Python 3.10+**

这是一个可扩展的“循环执行 Agent”项目骨架：核心库只使用 Python 标准库，支持持续迭代执行，直到满足 `done=True` 或命中停止条件（超时/最大步数/外部取消）。

## Python 版本要求

- Requires Python 3.10+
- Python 3.10 兼容说明：项目已避免使用 `dict[str, Any]`、`X | Y` 等仅 3.11+ 才支持的类型语法；CI 覆盖 3.10/3.11/3.12
- 若使用 Python 3.9 或更低版本，可能在导入阶段因类型语法不兼容报错（例如 `TypeError: 'type' object is not subscriptable`）

## 最短可执行路径（推荐默认）

```bash
python -m pip install -e .
python -m unittest discover -s tests -p "test_*.py" -v
```

## NPM 一键安装运行（Node 用户）

```bash
npm i -g git+https://github.com/t0ugh-sys/LoopAgent.git
loopagent tools
loopagent code --goal "inspect README then finish" --workspace . --provider mock --model mock-v3 --output json
```

> 注意：`npm i -g @t0ugh-sys/loopagent` 仅在包已发布到 npm registry 后可用；未发布会返回 404。

说明：

- 首次运行会自动检测 Python 3.10+、创建本地虚拟环境并安装当前包
- 可通过 `LOOPAGENT_PYTHON` 指定 Python 可执行文件

运行 CLI：

```bash
python -m loop_agent.cli --goal "write a one-line self introduction" --strategy demo --output json
```

Agent CLI（子命令）：

```bash
python -m loop_agent.agent_cli tools
python -m loop_agent.agent_cli code --goal "inspect README then finish" --workspace . --provider mock --model mock-v3 --output json
```

CI 使用 GitHub Actions 在 Python 3.11/3.12 上运行 `unittest`（见 `.github/workflows/ci.yml`）。

核心引擎在 `step` 抛异常时不会让进程直接崩溃，而是返回 `stop_reason=step_error` 并附带错误信息，便于上层统一治理。
核心引擎支持可选 `observer` 事件回调，便于接日志、埋点和监控系统。

## 依赖说明

- core（`src/loop_agent/`）是 stdlib-only
- `requirements.txt` 仅作为 examples/扩展依赖占位

## 结构

- `src/loop_agent/`：核心库（stdlib-only）
- `tests/`：单元测试（`unittest`）
- `examples/`：可选示例（可能需要额外依赖）

## 快速开始（安装与运行测试）

在项目根目录执行：

```bash
python -m pip install -e .
python -m unittest discover -s tests -p "test_*.py" -v
```

## 快速开始（运行 CLI）

```powershell
python -m loop_agent.cli --goal-file .\goal.txt --strategy demo
```

conda 等价命令：

```powershell
conda --no-plugins run --no-capture-output -n <your_env> python -m loop_agent.cli --goal-file .\goal.txt --strategy demo
```

内置策略：

- `demo`：固定三轮收敛
- `json_stub`：按 JSON 协议迭代，模拟 LLM 输出
- `json_llm`：调用可配置模型（provider/model 可切换）

例如切换到 `json_stub`：

```powershell
python -m loop_agent.cli --goal-file .\goal.txt --strategy json_stub --history-window 2
```

conda 等价命令：

```powershell
conda --no-plugins run --no-capture-output -n <your_env> python -m loop_agent.cli --goal-file .\goal.txt --strategy json_stub --history-window 2
```

切换模型（参数化切换）：

```bash
python -m loop_agent.cli --goal "answer in json protocol" --strategy json_llm --provider mock --model qwen-max
```

OpenAI-compatible provider 示例：

```bash
set OPENAI_API_KEY=sk-xxx
python -m loop_agent.cli --goal "answer in json protocol" --strategy json_llm --provider openai_compatible --model gpt-4o-mini --base-url https://api.openai.com/v1 --wire-api chat_completions
```

如果服务端是 Responses 协议（例如 `wire_api=responses`）：

```bash
set OPENAI_API_KEY=sk-xxx
python -m loop_agent.cli --goal "answer in json protocol" --strategy json_llm --provider openai_compatible --model gpt-5.3-codex --base-url https://codex-api.packycode.com/v1 --wire-api responses
```

若网关需要额外头（例如租户路由）：

```bash
python -m loop_agent.cli --goal "answer in json protocol" --strategy json_llm --provider openai_compatible --model gpt-5.3-codex --base-url https://codex-api.packycode.com/v1 --wire-api responses --provider-header "x-tenant:my-team" --provider-header "x-trace-id:demo-1"
```

网关不稳定时可开启重试与模型回退：

```bash
python -m loop_agent.cli --goal "answer in json protocol" --strategy json_llm --provider openai_compatible --model gpt-5.3-codex --fallback-model gpt-5-codex --base-url https://codex-api.packycode.com/v1 --wire-api responses --max-retries 3 --retry-backoff-s 1.0 --retry-http-code 502 --retry-http-code 503
```

Anthropic Claude provider 示例：

```bash
set ANTHROPIC_API_KEY=sk-ant-xxx
python -m loop_agent.cli --goal "answer in json protocol" --strategy json_llm --provider anthropic --model claude-3-opus-20240229
```

Google Gemini provider 示例：

```bash
set GEMINI_API_KEY=xxx
python -m loop_agent.cli --goal "answer in json protocol" --strategy json_llm --provider gemini --model gemini-pro
```

`code` 子命令同样支持 provider/model 切换：

```bash
set OPENAI_API_KEY=sk-xxx
python -m loop_agent.agent_cli code --goal "fix failing test" --workspace . --provider openai_compatible --model gpt-4o-mini --base-url https://api.openai.com/v1 --wire-api chat_completions
```

机器可读输出（便于 CI 或平台接入）：

```powershell
python -m loop_agent.cli --goal-file .\goal.txt --output json --include-history
```

事件落盘（JSONL）：

```powershell
$env:PYTHONPATH="src"
conda --no-plugins run --no-capture-output -n base python -m loop_agent.cli --goal-file .\goal.txt --observer-file .\events.jsonl
```

失败返回非零退出码（适合 CI）：

```powershell
$env:PYTHONPATH="src"
conda --no-plugins run --no-capture-output -n base python -m loop_agent.cli --goal-file .\goal.txt --strategy json_stub --max-steps 1 --exit-on-failure
```

基础运行记录（默认开启）：

- 每次 CLI 运行会在 `.loopagent/runs/<timestamp>/` 生成：
- `events.jsonl`：step/tool/stop 事件流
- `summary.json`：本次运行摘要

可通过 `--no-record-run` 关闭，或通过 `--runs-dir` 修改记录目录。

## 上下文记忆（默认开启）

- 默认记忆根目录：`.loopagent/runs`
- 每次运行会落盘到 `.loopagent/runs/<run_id>/`
- 关键文件：
- `events.jsonl`：完整事件流（可回放）
- `state.json`：当前快照（goal、step_index、last_output、history_tail）
- `summary.json`：长期摘要（goal/current_plan/facts/work_done/open_questions/next_actions）

CLI 参数：

- `--memory-dir`：记忆根目录（默认 `.loopagent/runs`）
- `--run-id`：指定运行 ID（默认 UTC 时间戳）
- `--summarize-every`：每 N 个事件更新一次摘要（默认 5）

每一步都会固定注入结构化上下文：

- `state_summary`（长期摘要）
- `last_steps`（最近 K 步，K 由 `--history-window` 控制）

示例 `goal.txt` 请用 UTF-8 保存，例如内容为：

```
给我一段 50 字以内的中文自我介绍
```

## 运行示例（JSON 协议 + stub 模型）

```powershell
python .\examples\json_loop_stub_demo.py
```

## 推荐用法（安装为包）

开源场景默认推荐上面的 `pip install -e .` 路径；下面是 Conda 环境里的等价写法：

```powershell
python -m pip install -e .
python -m unittest discover -s tests -p "test_*.py" -v
python -m loop_agent.cli --goal-file .\goal.txt
```

## Skill 系统

LoopAgent 支持可插拔的 Skill 系统，可以动态加载功能模块。

### 内置 Skills

| Skill | 说明 | 依赖 |
|-------|------|------|
| `web_search` | 联网搜索、获取网页 | stdlib |
| `memory` | 分析历史运行、学习模式 | stdlib |
| `files` | 文件读写、搜索 | stdlib |
| `commands` | 运行 Shell 命令 | stdlib |
| `browser` | 浏览器自动化 | playwright |

### 使用 Skill

```bash
# 加载指定 skills
python -m loop_agent.agent_cli code --goal "search for info" --skill web_search --skill memory

# 加载所有内置 skills（默认）
python -m loop_agent.agent_cli code --goal "your goal" --skill all
```

### 创建自定义 Skill

```python
from loop_agent.skills import Skill, register_skill

class MySkill(Skill):
    name = "my_skill"
    description = "My custom skill"
    
    def get_tools(self):
        def my_tool(args):
            return ToolResult(id="my_tool", ok=True, output="Hello!", error=None)
        return {"my_tool": my_tool}

# 注册并使用
register_skill(MySkill)
# 或通过 --skill 参数加载
```

### Skill 工具

- `web_search`: 搜索网络（query 参数）
- `fetch_url`: 获取网页内容（url 参数）
- `analyze_memory`: 分析历史运行（memory_dir, goal_filter, limit 参数）
- `read_file`, `write_file`, `apply_patch`, `search`: 文件操作
- `run_command`: 执行命令（command 或 cmd 参数）
- `browser_navigate`, `browser_click`, `browser_fill`, `browser_screenshot`: 浏览器操作
