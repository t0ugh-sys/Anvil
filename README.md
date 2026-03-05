# LoopAgent

这是一个可扩展的“循环执行 Agent”项目骨架：核心库只使用 Python 标准库，支持持续迭代执行，直到满足 `done=True` 或命中停止条件（超时/最大步数/外部取消）。

核心引擎在 `step` 抛异常时不会让进程直接崩溃，而是返回 `stop_reason=step_error` 并附带错误信息，便于上层统一治理。

## 结构

- `src/loop_agent/`：核心库（stdlib-only）
- `tests/`：单元测试（`unittest`）
- `examples/`：可选示例（可能需要额外依赖）

## 快速开始（运行测试）

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

例如切换到 `json_stub`：

```powershell
$env:PYTHONPATH="src"
conda --no-plugins run --no-capture-output -n base python -m loop_agent.cli --goal-file .\goal.txt --strategy json_stub --history-window 2
```

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

在 CI/企业项目里更推荐先安装为可编辑包，再运行测试/命令：

```powershell
conda --no-plugins run -n base python -m pip install -e .
conda --no-plugins run --no-capture-output -n base python -m unittest discover -s tests -p "test_*.py" -v
conda --no-plugins run --no-capture-output -n base python -m loop_agent.cli --goal-file .\goal.txt
```
