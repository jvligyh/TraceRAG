# 图片与扫描PDF识别

TraceRAG 0.4.0支持在本机识别中英文印刷文字，并将识别结果送入原有的切块、检索、回答和引用链路。使用RapidOCR 3.10，文字检测与识别模型为PP-OCRv6-small，ONNX Runtime在CPU运行。OCR不依赖DeepSeek，也不需要另一个API Key；使用DeepSeek回答时，选中的识别文本会作为证据发送给模型。

## 日常操作

1. 双击项目根目录的`启动TraceRAG.cmd`。
2. 上传PNG/JPEG/WebP/BMP/单帧TIFF图片，或扫描PDF。可以多选文件，界面逐个处理；首次识别需下载并加载模型，请等待上传结果。
3. 查看每份文件成功生成识别文本的OCR页数和警告，在文档库勾选要查询的文件，再提问。该页数不包含未识别到可用文字的空白页，也不等于PDF总页数。
4. 对照引用中的识别文本，点击原图或PDF页码链接核对数字、单位和专业词。

SVG、GIF及多帧图片不在当前支持范围。多帧TIFF请拆成单张图片或转换成PDF。文件后缀不能代替内容校验，损坏图片会被拒绝。

## PDF如何决定使用OCR

每页先提取原生文字层，归一化后去掉空白字符。如果不足20个字符，则将这一页渲染成图片并尝试OCR；达到20个字符的页保留原生文字，不再OCR。这样同一PDF可以同时包含原生文字页和扫描页。

这是一个实用的回退规则，不能自动识别所有混合版面。文字足够多的页中即使有一幅带文字的插图，也不会单独OCR那幅图。需要查询插图中的文字时，可将其裁出并作为图片上传。当前也没有从图形关系、曲线、坐标或图例推导结论的图表语义能力。

扫描页识别后保留PDF实际物理页码，空白页或识别为空的页不会让后续页码前移。上传结果会报告未建立索引的页；如果OCR没有更多文字而原生页仍有可用短文本，会保留短文本并提示。图片没有PDF页码，引用直接打开原图。

## 配置与资源

默认启用OCR，配置在项目`.env`中，外部环境变量优先。修改后重启服务；解析文本是上传时的快照，旧文档不会自动重新识别。需要比较新配置时，可使用独立数据目录，或删除对应旧记录后重新上传。

| 配置 | 默认值 | 含义 |
|---|---|---|
| `TRACERAG_OCR_ENABLED` | `true` | 关闭后不识别图片或扫描页 |
| `TRACERAG_OCR_MAX_PAGES` | `30` | 每份PDF中需要OCR的最大页数，不是PDF总页数 |
| `TRACERAG_OCR_MAX_IMAGE_PIXELS` | `20000000` | 单张图片或渲染页最多2000万像素 |
| `TRACERAG_OCR_MIN_CONFIDENCE` | `0.5` | 过滤低于此分数的识别行 |

网页/API单文件固定上限10 MiB，服务配置可进一步收紧，调大`TRACERAG_MAX_UPLOAD_BYTES`不能放宽HTTP上限。超出待OCR页数上限会要求拆分PDF，不会只识别前30页并假装完整。PDF默认以2倍页面尺寸渲染，渲染后的像素也受上限约束。大图请先缩小；多份文件上传仍逐份显示完成或错误。

服务的OCR模型缓存位于运行数据目录同级的`models/ocr`，本项目默认为`data/models/ocr`。本机已下载约32 MB模型；首次需网络下载，后续从缓存加载，进程内复用。OCR使用CPU，`TRACERAG_DEVICE=auto/cuda`只影响相应的Transformers模型，不会把OCR改为CUDA，也不会替换当前PyTorch安装。

`/health`中的`ocr_enabled`表示配置已启用，`ocr_loaded`表示引擎已加载。首次识别前`ocr_loaded=false`是正常状态。安装缺失时，在项目虚拟环境执行：

```powershell
python -m pip install -c constraints.txt -e .
```

文字检测、方向处理、识别和PDF渲染都在本地运行；OCR步骤不会把原始图片上传给外部识别服务。后续若选DeepSeek作答，问题和检索选中的证据文本仍按原有生成流程发送。故障处理见[运行指南](runbook.md)。

## 如何理解识别结果

文档、Chunk及引用保留`extraction_method`，取值为`native`或`ocr`；OCR结果另有`ocr_confidence`。界面显示“OCR识别文本，请核对原图”，并明确置信度不等于正确率。该置信度是被保留识别行的模型分数平均值，不是单字正确概率，也不是答案的事实正确率。检索分数同样只是排序依据。

OCR可能漏字或误认数字、负号、小数点、上下标、单位及专业符号。调低阈值可能保留更多噪声，调高则可能漏掉有用文字。当前没有复杂公式、表格单元格或多栏阅读顺序的完整布局还原，也未对手写、低清照片或各专业文献建立可靠性保证；需要回到原图核对。

## 已完成的真实验证

在本机用真实OCR和DeepSeek处理两份固定合成印刷实验记录，随后用Edge完成界面操作。这两份样例用于核对链路和页码，不是实际科研实验，也不是OCR准确率benchmark：

| 样例 | 核对结果 |
|---|---|
| `OCR中英实验记录.png` | 原图中光电转换效率为23.5%；识别文本与回答均返回23.5%，引用可以打开原PNG |
| `OCR扫描实验记录.pdf` | 第2页为空并显示未建索引警告；催化剂温度为42 C，回答引用物理第3页，原文链接保留`#page=3` |

原文件下载字节与上传一致。浏览器验收还核对了OCR标注、置信度非正确率提示、新标签页原图、PDF页码链接，以及390px布局无横向溢出、无页面脚本错误。具体记录：

- [上传、识别、回答与原文件校验](evidence/ocr-live-validation.json)
- [重启后检索、原文件保留和空白图拒绝](evidence/ocr-restart-validation.json)
- [Edge浏览器验收JSON](../results/ocr-browser/browser-validation.json)
- [图片问答截图](../results/ocr-browser/image-answer.png)、[扫描PDF问答截图](../results/ocr-browser/scan-answer.png)、[手机布局截图](../results/ocr-browser/mobile.png)

重启后，保存的识别文本仍可检索，原文件和物理页码保留，无需为了查询旧记录重新OCR；空白图片返回400、`ocr_empty`且不写入文档库。本轮全套157项单元/集成测试通过，另通过`pip check`和源码编译检查。

`results/ocr-browser`是本机实际验收目录，含录屏，受Git忽略；仅复制源码时这些文件可能不在其中。上述两个数值识别成功不代表对其他文档、手写或复杂场景已测得准确率。

## 开源来源与许可证

引擎来自[RapidOCR官方仓库](https://github.com/RapidAI/RapidOCR)，使用方式见[官方文档](https://rapidai.github.io/RapidOCRDocs/main/en/install_usage/rapidocr/usage/)。模型来自PaddleOCR，由RapidOCR转换或分发；PP-OCRv6-small的检测、识别模型及方向分类模型的来源与SHA256见[官方模型许可与归属声明](https://rapidai.github.io/RapidOCRDocs/main/en/model_licenses/)。

RapidOCR代码及该声明覆盖的模型采用Apache-2.0，但代码与上游模型的著作权归属不同。分发模型时需保留相应许可证、归属和修改说明；以[RapidOCR LICENSE](https://github.com/RapidAI/RapidOCR/blob/main/LICENSE)、[模型许可声明](https://rapidai.github.io/RapidOCRDocs/main/en/model_licenses/)及[PaddleOCR LICENSE](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE)为准。开源许可不构成识别准确率或特定用途保证。
