# LoopAgent

这是一个可扩展的“循环执行 Agent”项目骨架：核心库只使用 Python 标准库，支持持续迭代执行，直到满足 `done=True` 或命中停止条件（超时/最大步数/外部取消）。

## Python 版本要求（必读）

- Requires Python 3.11+
- 若使用 Python 3.8/3.9/3.10，可能在导入阶段因 `dict[str, Any]` 等类型语法报错（例如 `TypeError: 'type' object is not subscriptable`）

## 最短可执行路径（推荐默认）

```bash
python -m pip install -e .
python -m unittest discover -s tests -p "test_*.py" -v
```

运行 CLI：

```bash
python -m loop_agent.cli --goal "write a one-line self introduction" --strategy demo --output json
```

OpenClaw-style CLI（子命令）：

```bash
python -m loop_agent.openclaw_cli tools
python -m loop_agent.openclaw_cli code --goal "inspect README then finish" --workspace . --provider mock --model mock-v3 --output json
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

## 可选：Conda 方式

在项目根目录执行：

```powershell
$env:PYTHONPATH="src"
conda --no-plugins run --no-capture-output -n base python -m unittest discover -s tests -p "test_*.py" -v
```

## 快速开始（运行 CLI）

```powershell
$env:PYTHONPATH="src"
conda --no-plugins run --no-capture-output -n base python -m loop_agent.cli --goal-file .\goal.txt --strategy demo
```

内置策略：

- `demo`：固定三轮收敛
- `json_stub`：按 JSON 协议迭代，模拟 LLM 输出
- `json_llm`：调用可配置模型（provider/model 可切换）

例如切换到 `json_stub`：

```powershell
$env:PYTHONPATH="src"
conda --no-plugins run --no-capture-output -n base python -m loop_agent.cli --goal-file .\goal.txt --strategy json_stub --history-window 2
```

切换模型（OpenClaw-style 参数化切换）：

```bash
python -m loop_agent.cli --goal "answer in json protocol" --strategy json_llm --provider mock --model qwen-max
```

OpenAI-compatible provider 示例：

```bash
set OPENAI_API_KEY=sk-xxx
python -m loop_agent.cli --goal "answer in json protocol" --strategy json_llm --provider openai_compatible --model gpt-4o-mini --base-url https://api.openai.com/v1
```

OpenClaw-style `code` 子命令同样支持 provider/model 切换：

```bash
set OPENAI_API_KEY=sk-xxx
python -m loop_agent.openclaw_cli code --goal "fix failing test" --workspace . --provider openai_compatible --model gpt-4o-mini --base-url https://api.openai.com/v1
```

机器可读输出（便于 CI 或平台接入）：

```powershell
$env:PYTHONPATH="src"
conda --no-plugins run --no-capture-output -n base python -m loop_agent.cli --goal-file .\goal.txt --output json --include-history
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

- 每次 CLI 运行会在 `runs/<timestamp>/` 生成：
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
$env:PYTHONPATH="src"
conda --no-plugins run --no-capture-output -n base python .\examples\json_loop_stub_demo.py
```

## 推荐用法（安装为包）

开源场景默认推荐上面的 `pip install -e .` 路径；下面是 Conda 环境里的等价写法：

```powershell
conda --no-plugins run -n base python -m pip install -e .
conda --no-plugins run --no-capture-output -n base python -m unittest discover -s tests -p "test_*.py" -v
conda --no-plugins run --no-capture-output -n base python -m loop_agent.cli --goal-file .\goal.txt
```
