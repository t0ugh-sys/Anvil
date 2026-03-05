# Project Agent Rules (LoopAgent)

## 语言与编码
- 统一用中文沟通
- 文件使用 UTF-8

## 代码约定
- Python 代码必须包含 type hints
- 核心库（`src/`）不引入第三方依赖，仅用标准库
- 可选集成（如 LangChain/OpenAI）放在 `examples/`，并在文档中标注额外依赖

## 测试
- 使用 `unittest`
- 新增功能必须配套测试

