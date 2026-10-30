# 项目交付边界（2026-10-29）

当前版本0.4.0新增本地RapidOCR 3.10 / PP-OCRv6-small：支持PNG/JPEG/WebP/BMP/单帧TIFF及逐页扫描PDF回退，OCR文本进入原有检索/生成链路，保留来源、实际PDF页码、原文件与识别标注。默认限制为30个待OCR页、2000万像素和HTTP单文件10 MiB。真实合成实验记录经过OCR、DeepSeek和Edge验收，正确返回23.5%和第3页42 C；这是链路验证，不是OCR准确率benchmark。详见[OCR说明](ocr.md)、[实测JSON](evidence/ocr-live-validation.json)和[浏览器证据](../results/ocr-browser/browser-validation.json)。

本轮157项单元/集成测试、依赖一致性和源码编译检查通过；重启后OCR文本可检索、原文件与物理页码保留，空白图片返回400且不入库，见[重启验证](evidence/ocr-restart-validation.json)。

0.3.0日常使用补充：原文件持久化/旧库无损迁移、按上传文档过滤检索和工具调用、会话历史与独立检索问题改写、引用原PDF页码链接、多文件上传、浏览器刷新恢复、双击启动与端口识别。19页真实PDF经过实际上传、连续提问、引用页原文校对、浏览器原文打开和服务重启持久化验证；证据见real-paper-acceptance.json、daily-restart-validation.json与results/daily-real。

已实现本地完整链路：PDF/Markdown/TXT/HTML/图片 → 原生文字提取或OCR → 稳定切块 → SQLite持久化 → 双向翻译+BM25/Dense → RRF → 可选Cross-Encoder → DeepSeek → Chunk引用校验 → 网页/API。

提供增量向量缓存、模型常驻、search/read_chunk有界工具循环、超时/重试/参数校验/错误映射、数据库恢复、单元和HTTP集成测试、Docker/Compose/启动脚本、25篇实际公开语料和40条带出处种子题、可重跑消融与性能脚本。原Day1～Day5学习材料保留。

验收结果以docs/evidence实际文件为准。测试通过不代替真实模型验收；Dockerfile存在不表示Docker已经运行。

此前0.2.0验收完成单元/集成测试、12组检索消融各40题、DeepSeek端到端36条可答题返回引用与4条无答案题拒答，以及桌面/手机浏览器验收和录屏。后续版本增加文档使用与OCR测试，当前测试结果以实际测试输出为准。公开资料25篇为摘要或官方文档摘录，另有3篇中文笔记；问答库额外预载演示文档。人工事实审核仍待完成。

需要用户完成：独立复核40题相关性（human_reviewed目前false）、补真实问题留出集、逐条检查答案事实支持度、使用本人研究材料录制演示、理解代码并改写简历/模拟面试/投递。AI不能代替这些事情并声称人工完成。

未包含：图表语义理解、复杂表格公式版面还原、ANN/向量数据库、分布式索引、多租户/登录、公网部署。PDF原生文字达到20个非空白字符的页不会OCR其中的插图；OCR不能保证专业符号、数字和单位正确，识别分数不等于正确率。这些边界不能写成已解决。

原有未提交改动已保留，本次代码也在工作区，没有自动创建远程仓库或推送。可通过本地diff审阅。
