# TraceRAG

TraceRAG 是一个面向技术文档的「可追溯、评测优先」RAG 与工具调用服务。项目目标是准备华为 AI 应用工程师实习岗，而不是再做一个只能演示的 PDF 聊天机器人。

## 当前状态

Day 1 工程准备已完成，目前已有：

- 与模型供应商解耦的 `LLMClient` 接口；
- 确定性 Mock 模型客户端；
- 最小问答 Pipeline 和来源 ID 传递；
- 2 篇本地演示文档和 25 篇公开技术资料清单；
- 单元测试、冒烟脚本和接口文档；
- 第一版架构图和 15 天交付计划；
- 本地 `.venv`，TraceRAG 已以可编辑模式安装。
- 通过 OpenAI-compatible SDK 成功调用 DeepSeek 真实模型。

Mock 路径不需要 API Key；DeepSeek 路径仅在运行时读取密钥。仓库不保存真实凭据。

## 运行

```powershell
cd "D:\Desktop\实习\TraceRAG"
.\.venv\Scripts\Activate.ps1
python scripts\day1_smoke.py
python -m unittest discover -s tests -v
```

运行 DeepSeek 真实调用（脚本会隐藏密钥输入）：

```powershell
python scripts\day1_live.py
```

在新机器重建环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

如果只有 Windows Python Launcher，可将第一条改为 `py -3.11 -m venv .venv`。

## 目标链路

```text
PDF/Markdown/HTML -> 清洗/切块 -> BM25 + 稠密检索
                  -> RRF 融合 -> 重排 -> LLM 作答
                  -> 块级引用 -> 离线评测
```

## 工程原则

1. 简历中的每项声明都必须有可运行代码或实测结果支撑。
2. AI 生成代码必须通过人工理解、测试和验收标准。
3. 不向仓库提交任何供应商凭据。
4. 评测与功能同步建设，而不是演示完成后再补。

详见 `docs/architecture.md`、`docs/interfaces.md`、`docs/day1.md` 和 `docs/15-day-plan.md`。
