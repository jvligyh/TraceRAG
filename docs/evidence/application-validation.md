# 本地应用验收（2026-10-29）

当前版本0.4.0增加图片与扫描PDF的真实OCR验收，见下节。当前测试结果以运行输出为准，下面的旧版本记录保留原验收背景。

0.3.0日常验收：19页完整公开PDF上传后，单文档查询和代词追问均返回对应页码引用；未提供的确切耗电量拒答。原文件SHA256与下载PDF一致，实际服务重启后资料/原文件/限定范围检索仍正常。浏览器真实DeepSeek多轮、原文新标签页、刷新恢复scope/history、390px布局通过，详见real-paper-acceptance.json、daily-restart-validation.json和results/daily-real。

## 0.4.0 OCR真实验收

使用RapidOCR 3.10 / PP-OCRv6-small在本机CPU实际识别合成的中英文印刷实验记录，再通过真实DeepSeek问答与Edge浏览器核对。样例是为验证功能生成的固定文本，不是实际研究结果，也不是OCR准确率、手写或场景识别benchmark。

- `OCR中英实验记录.png`：识别出光电转换效率23.5%，限定该图片提问返回23.5%；引用带`extraction_method=ocr`、置信度和原图URL；原图字节与上传一致。
- `OCR扫描实验记录.pdf`：中间空白的物理第2页未建立索引，上传警告如实显示；温度答案为42 C，引用仍指向物理第3页，原文URL带`#page=3`。
- Edge实际提交两次问题，检查OCR标注、置信度“非正确率”提示、原图新标签页和PDF第3页链接；390px布局无横向溢出，未出现页面脚本错误。

API/上传及回答记录：[ocr-live-validation.json](ocr-live-validation.json)。浏览器记录、截图和录屏：[browser-validation.json](../../results/ocr-browser/browser-validation.json)、[图片回答](../../results/ocr-browser/image-answer.png)、[扫描PDF回答](../../results/ocr-browser/scan-answer.png)、[手机布局](../../results/ocr-browser/mobile.png)。`results`为本机验收产物，受Git忽略，源码副本中不保证附带。功能边界、配置和官方许可证见[OCR说明](../ocr.md)。

本轮157项单元/集成测试、`pip check`和源码编译检查通过。[ocr-restart-validation.json](ocr-restart-validation.json)记录了真实服务重启后OCR文本检索、原文件及物理页码保留；空白图片返回400、`ocr_empty`，且没有写入文档库。

## 此前0.2.0已执行

- HTTP真实服务127.0.0.1:8000，预载25篇公开资料、3篇中文笔记和1篇演示文档，共29份/116个Chunk。
- DeepSeek端到端40题：36题status=answered，4条无答案题status=insufficient_evidence；40次HTTP200。引用ID均属于传给模型的证据。详情answer-evaluation.json；这不是事实正确率或独立人工评测。
- DeepSeek真实ToolAgent：先search再final，返回已观察Chunk引用；见agent-live.json和runtime-validation.json。
- Edge/Playwright桌面1280px与手机390px：真实提交问题、得到答案和引用；无pageerror、无横向溢出。截图和录屏在results/browser（Git忽略）。
- Python源码compileall、112项unittest全部通过、pip check无依赖冲突；完整项目构建为dist/tracerag-0.2.0-py3-none-any.whl，确认包含web.html。
- 模型常驻与索引复用：runtime-validation中预热同问题的search-only total约0.006秒。它同时命中翻译缓存，不等于未见问题或含API生成的端到端延迟。
- MiniLM模型固定commit，ONNX SHA256记录在final-evaluation.json，包/硬件版本在environment.json。
- 12组检索/翻译/Embedding/重排组合全部完成40题；完整结果final-evaluation.json/md。Qwen Embedding与BGE重排均经过完整官方权重SHA256校验和真实CUDA推理；model-benchmark.json保存各自预热延迟与显存。

## 未执行或需人工

- 当前机器无docker命令；Compose配置通过YAML解析，未进行容器构建与运行，不能声称Docker已验收。
- CI工作流文件已提供，但没有远程GitHub执行记录。
- 40题是AI起草并核对证据字符串的种子集；人工相关性、漏标、事实支持度审查仍待完成。
- 本地应用无鉴权，不作为公网生产服务。
