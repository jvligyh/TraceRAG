# TraceRAG 0.4.0 快速开始

日常入口：双击`启动TraceRAG.cmd`，或执行`.\start.ps1 -OpenBrowser`。支持文档范围、连续对话、打开原文，以及本地OCR识别图片与扫描PDF。操作见[daily-use.md](daily-use.md)，识别边界见[ocr.md](ocr.md)。原有命令行入口保持兼容。

Python 3.11+；本机验证环境 Windows / Python 3.12 / RTX 4060 Laptop。

## 启动

当前机器环境已经安装，在PowerShell执行：

```powershell
cd D:\Desktop\实习\TraceRAG
.\start.ps1
```

浏览器 http://127.0.0.1:8000，Swagger http://127.0.0.1:8000/docs。上传PDF/Markdown/TXT/HTML或PNG/JPEG/WebP/BMP/单帧TIFF。扫描PDF逐页自动回退OCR，首次识别需要加载OCR模型。解析完成后，首次查询惰性建索引；后续复用模型和索引，重启复用磁盘向量缓存。

新机器安装：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -c constraints.txt -e ".[local-translation]"
# NVIDIA驱动兼容CUDA13时，替换为官方CUDA wheel：
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
```

已有.env不会被覆盖，参考.env.example配置。DeepSeek密钥只从环境读取。`TRACERAG_PROVIDER=deepseek`使用真实生成；`extractive`返回证据摘录，明确不属于模型生成。没有密钥默认摘录模式。

无需下载模型的验收配置：

```powershell
$env:TRACERAG_RETRIEVAL="bm25"
$env:TRACERAG_TRANSLATOR="none"
$env:TRACERAG_PROVIDER="extractive"
$env:TRACERAG_OCR_ENABLED="false"
.\start.ps1
```

该配置用于文字层PDF或文本文件，关闭OCR后不能识别图片和扫描页。这些临时变量只作用于当前PowerShell；恢复默认配置请开新终端或移除覆盖。

## 常用配置

| 变量 | 默认/选择 |
|---|---|
| TRACERAG_RETRIEVAL | hybrid；可选bm25/dense |
| TRACERAG_TRANSLATOR | qwen；可选none/deepseek/marian，Marian仅中译英 |
| TRACERAG_EMBEDDING | minilm；可选qwen，使用固定revision Qwen3-Embedding-0.6B |
| TRACERAG_RERANKER | none；bge启用BGE Cross-Encoder |
| TRACERAG_DEVICE | auto；可选cpu/cuda |
| TRACERAG_PROVIDER | 有密钥时deepseek，否则extractive |
| TRACERAG_DATA_DIR | data/runtime |
| TRACERAG_TIMEOUT_SECONDS | 单次API超时60秒 |
| TRACERAG_API_RETRIES | SDK重试2次 |
| TRACERAG_OCR_ENABLED | true；本地OCR开关 |
| TRACERAG_OCR_MAX_PAGES | 30；一份PDF中需要OCR的最大页数 |
| TRACERAG_OCR_MAX_IMAGE_PIXELS | 20000000；单张图片/渲染页像素上限 |
| TRACERAG_OCR_MIN_CONFIDENCE | 0.5；保留识别行的最低分数，不是正确率 |
| TRACERAG_MAX_UPLOAD_BYTES | 默认10485760；可收紧上传限制，网页/API固定上限仍为10 MiB |

MiniLM使用固定ONNX revision、mean pooling、CPU推理。Qwen翻译/Embedding/BGE使用Transformers，可CUDA推理。8GB显存机器可先使用MiniLM+Qwen翻译，按消融结果决定是否同时启用更大Embedding/重排。

OCR使用RapidOCR 3.10的PP-OCRv6-small，经ONNX Runtime在CPU运行，`TRACERAG_DEVICE`不改变OCR设备。默认模型缓存为`data/models/ocr`，本机已下载约32 MB；首次需要网络，缓存后本地识别。PDF一页原生文字去掉空白后不足20个字符才回退OCR，已有文字页内的插图不会单独识别。OCR引用保留实际PDF页码，并可打开原图核对。详见[OCR配置与验证](ocr.md)。

## 命令行

```powershell
.\.venv\Scripts\python.exe -m tracerag.cli ingest data\demo\rag_evaluation.md data\demo\traceability.md
.\.venv\Scripts\python.exe -m tracerag.cli query "为什么要分别评估检索和生成？"
.\.venv\Scripts\python.exe -m tracerag.cli query "What is traceability?" --language Chinese --search-only
.\.venv\Scripts\python.exe -m tracerag.cli chat
.\.venv\Scripts\python.exe -m tracerag.cli agent "为什么要分别评估检索和生成？" --max-steps 6
```

`chat`连续问答复用模型。旧Day1～Day5脚本仍可单独学习使用。

## 测试与实测

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\evaluate.py
.\.venv\Scripts\python.exe scripts\verify_runtime.py --live --agent --output docs\evidence\runtime-validation.json
.\.venv\Scripts\python.exe scripts\benchmark_models.py --help
```

25篇实际公开语料（21篇摘要、4篇官方文档摘录）及来源hash在data/corpus。另有3篇AI中文笔记。40条带证据种子题已程序核对，但仍需独立人工相关性复核。摘要不等于全文，引用ID合法不等于事实全部受支持。完整消融见evaluation.md。

OCR的真实合成样例与浏览器验收分别记录在[ocr-live-validation.json](evidence/ocr-live-validation.json)和[results/ocr-browser](../results/ocr-browser/browser-validation.json)，验证了23.5%和PDF第3页42 C的识别、问答与引用。这些记录不代表一般文献、手写或复杂场景的OCR准确率。

## Docker

```powershell
docker compose up --build -d
```

CPU镜像、非root用户、本机8000端口、具名volume保留文档/模型。若本机无Docker，容器构建/运行属于待环境验收事项。
