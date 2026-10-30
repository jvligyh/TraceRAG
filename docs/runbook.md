# 运行与故障排查

项目目录执行start.ps1，网页 http://127.0.0.1:8000。启动不立即下载所有模型，首次查询可能下载/加载当前选中模型。上传后惰性建索引，同一个进程连续查询复用模型。Ctrl+C停止后重启，文档和向量缓存仍保留。

默认数据库data/runtime/tracerag.sqlite3。备份时停止服务并复制整个data/runtime。0.3.0新上传文件同时保存原始字节、归一化文档/切块和来源名；引用可打开原PDF并定位页码。旧版记录未保存原文件时需要重新上传。删除文档删除原文件与关联文档/切块，向量内容缓存可能保留供同文本复用；该缓存仅本地存储。

解析与切块结果是上传时保存的快照。修改chunk_chars/overlap或OCR设置后，要在网页删除对应旧文档并重新上传，或使用新的TRACERAG_DATA_DIR做独立实验；重启服务不会自动用新参数改写旧文档。Embedding配置改变则会使用新的模型指纹缓存。

DeepSeek生成会发送问题和选定证据文本，Qwen本地翻译不向模型服务发送问题。密钥不写入日志和响应。当前服务没有登录/多租户隔离，仅绑定127.0.0.1，不要直接作为公网服务。

环境变量优先于.env。provider=deepseek缺少密钥时回答端点返回503；extractive为原文摘录，不能当成LLM回答。旧mock用于教学脚本，不属于service provider。翻译模式deepseek也需要密钥。

device=auto有CUDA时自动选择，否则CPU。Qwen翻译、Qwen Embedding、BGE同时常驻会累加显存；OOM时关闭reranker、改CPU或用MiniLM后重启。多个服务进程会分别加载模型，因此固定单worker，模型调用受锁保护。

API单次超时默认60秒、重试2次，所以总耗时可能超过60秒。HTTP校验400/422，文档不存在404，超限413，模型暂不可用503。日志记录request ID、状态与耗时，不打印问题和全文。

MiniLM采用固定repo commit与mean pooling，Qwen/BGE使用固定revision。正式评测需同时保存语料hash、题目hash、运行包版本、设备和模型文件hash。安装依赖范围不等于未来全部环境性能一致。

Embedding和翻译模型保存在Hugging Face持久缓存，可通过HF_HOME指定；Docker为/app/data/models。Windows无符号链接时可能额外占盘。下载失败应重试，不会把模型错误伪装成正常空榜。HTML仅抽取UTF-8可见文本，不执行脚本、不抓远端资源。

## OCR上传与排查

0.4.0默认启用RapidOCR 3.10 / PP-OCRv6-small，本地ONNX Runtime CPU推理，不依赖CUDA版PyTorch。支持PNG/JPEG/WebP/BMP/单帧TIFF；PDF优先提取文字层，一页去掉空白后不足20个字符时才渲染并OCR。原生文字达到20个字符的页不会识别页内插图；需识别其中的文字时，可裁图单独上传。当前不做图表语义理解或复杂公式布局还原。

OCR模型单独缓存在应用数据目录的同级`models/ocr`，本项目默认是`data/models/ocr`，本机已下载约32 MB。首次识别会下载/加载，之后进程内复用；不要因等待而重复提交。`/health`的`ocr_enabled`表示配置启用，`ocr_loaded`表示已加载，后者在首次识别前为false是正常状态。仅使用文字层PDF或文本文件无需加载OCR。

默认限制与处理方式：

| 情况 | 处理 |
|---|---|
| HTTP单文件超过10 MiB | 缩小或拆分文件；网页/API固定上限10 MiB，`TRACERAG_MAX_UPLOAD_BYTES`只能进一步收紧 |
| 单图或PDF渲染页超过2000万像素 | 缩小图片或PDF页面尺寸；`TRACERAG_OCR_MAX_IMAGE_PIXELS`默认20000000 |
| 一份PDF超过30页需要OCR | 拆分PDF；`TRACERAG_OCR_MAX_PAGES`默认30，统计需OCR页数而非总页数 |
| 提示OCR关闭 | 设置`TRACERAG_OCR_ENABLED=true`后重启 |
| 提示依赖缺失 | 在项目虚拟环境执行`python -m pip install -c constraints.txt -e .` |
| 初始化/模型下载失败 | 检查网络、磁盘权限和`data/models/ocr`；修复后重试，错误不会作为正常识别结果 |
| 未识别到可用文字或某页为空 | 查看上传警告；清晰度不足时更换扫描，空白页不建索引，后续引用仍保留原始页码 |

低于`TRACERAG_OCR_MIN_CONFIDENCE`的识别行会被过滤，默认阈值0.5。证据上的OCR置信度是保留行分数的平均值，不是正确率；较高分数仍可能误认数字、符号或单位。上传警告和引用标注用于提醒原文核对，不能据此声称OCR经过准确率评测。具体配置、许可证与真实验收见[ocr.md](ocr.md)。

演示顺序：上传data/demo两篇文档→中文问分别评估原因→检查引用→英文问并选中文答→只检索/工具模式查看trace→重启查询验证持久化→询问无证据问题检查拒答。Dense返回最近邻不代表资料足以回答。
