# 评测与语料使用说明

端到端回答验收：服务运行后执行`python scripts/evaluate_answers.py`，保存每题答案/引用/HTTP状态，分别报告引用ID合法比例和无答案拒答率。它不自动判断事实正确性，`human_answer_review=pending`的记录应由人逐条核对。`scripts/verify_runtime.py --live --agent`为小样例真实模型冒烟；浏览器验收脚本`scripts/browser_smoke.cjs`需要Playwright和本机Edge，截图/录屏保存到被Git忽略的results/browser。

仓库包含25篇实际下载的公开资料：21篇 arXiv 论文摘要和4篇官方文档摘录。`data/corpus/manifest.json` 记录原始URL、实际重定向URL、抓取时间、正文长度、文件 SHA256、稳定 document_id。摘要不等于论文全文，当前结果不支持对完整文献处理能力的结论。另有3篇明确注明 AI 起草的中文阅读笔记，用于双语检索验收，不能冒充官方译文。

`data/evaluation/questions.jsonl` 包含40条种子题：36条可回答题、4条无答案题。每条有 source_id、绑定到当前正文哈希的 relevant_document_ids、证据原文和语言/术语/数字/推理标签。运行时验证这些ID仍与语料一致，正文改变就要求重新检查标注；完整结果JSON记录每条题目的相关ID与返回ID。证据字符串逐条通过程序核对，`human_reviewed` 当前全部为 false。这是AI起草并检查出处的草稿，不是人工标注集。

英文问题检索中文笔记的题同时把英文原始摘要列为相关文档，避免把事实相同的英文原文错误判成不相关。因此这些题属于双语证据测试，不是纯粹的“只允许中文结果”的跨语言基准。标注多篇相关文档时，Recall@1 不会因为命中其中一篇而等于1；Hit Rate@1会。两种指标均在JSON中给出。

## 复现

在项目虚拟环境内运行：

```powershell
python scripts/fetch_corpus.py
python scripts/evaluate.py
```

默认执行词级BM25、字符BM25、MiniLM Dense、无查询翻译的混合RRF，输出 `docs/evidence/final-evaluation.json` 和同名 Markdown。已下载语料默认复用；加 `--refresh` 才重新抓取，更新正文后应重新检查证据标注。

完整模型对照（模型首次下载需要网络；DeepSeek会调用已有API并产生用量）：

```powershell
python scripts/evaluate.py `
  --strategies bm25,char,dense,hybrid-none,lexical-none,lexical-qwen,hybrid-qwen,hybrid-marian,hybrid-deepseek,dense-qwen,hybrid-qwen-embedding,hybrid-rerank `
  --output docs/evidence/full-ablation.json
```

`hybrid-rerank` 在MiniLM + Qwen翻译 + RRF后对前20个Chunk用BGE重排，可通过 `--rerank-depth` 固定候选深度。需要先完成模型下载。各组开始前清除查询翻译的结果缓存，使延迟包含真实翻译；模型本身允许常驻，加载/预热时间另外报告。

分批运行时可用 `--append` 追加策略；程序只允许相同语料hash、问题hash、依赖版本、切块和候选配置的报告合并，避免把不同实验条件的分数放在一张表中。DeepSeek的线上模型名称属于供应商别名，报告不能冻结供应商内部权重，应连同实测时间一起解读。

快速诊断可以加 `--limit 5`，不能把限制后的结果当作40题全量结果。没有运行过的组合没有分数；失败组记录失败类型并以非零退出码提示，不填造假的占位成绩。

## 指标口径

- Recall@1/3/5：去重后的文档榜单中，命中的相关文档数除以所有已标注相关文档数。
- Hit Rate@1/3/5：前k个文档是否至少命中一篇，适合看“有没有找到证据”。
- MRR@5：第一篇相关文档名次的倒数，对可回答问题取平均，仅看前5。
- 无答案题：单独记录空检索比例，不纳入Recall/MRR。Dense一定返回最近邻，空检索比例不能代替生成拒答准确率。
- 引用ID有效率：只检查引用ID是否属于供给生成模型的证据。ID合法不表示句子受证据支持，也不表示答案正确。
- 延迟：加载与建索引、单独预热、各题查询耗时分开，报告平均值/P50/P95。每题只执行一遍；查询集合很小，P95只能作本机诊断。

每组固定相同正文、Chunk、问题和 `candidate_k`。报告包含语料hash、问题hash、包版本、切块参数和模型名称。MiniLM使用FastEmbed的mean pooling，并固定到 `qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q` 的 `faf4aa4225822f3bc6376869cb1164e8e3feedd0`；报告另记录本机ONNX文件的SHA256。查询翻译Qwen固定revision。不能跨模型比较平均余弦分数来判断谁更好。

可选 `hybrid-marian` / `lexical-marian` 是单向中译英基线：英文问题保留原文，不生成中文变体；中文问题才调用Marian。完整双向能力仍由Qwen和DeepSeek提供。`lexical-none` / `lexical-qwen` 等可在固定BM25上隔离查询翻译收益。

## 人工验收

请逐条核对问题是否自然、证据是否确实回答问题、是否还有漏标的相关资料；确认后再将对应 `human_reviewed` 改成 true。尤其检查跨语言/中文笔记题、多个相关文档题和无答案题。另准备独立于开发过程的真实论文与用户问题作为留出集，避免只在开发样例上优化。

当前自动运行不评判生成答案的事实正确性，也没有使用另一个LLM给自己的答案打分后声称人工正确率。端到端回答质量还需记录每题的答案、引用、拒答，并由人核对。

## 2026-10-29 全量12组实测与选型

12组均已实际运行同一份40题、28篇摘要/文档/笔记、115个Chunk，无失败组。下列Recall/MRR只对36条可回答题计算；4条无答案题单独保留。40条全部仍是 `human_reviewed=false` / `human_pending`，这些数字只能用于开发期对照，不能写成“人工评测准确率”或完整论文泛化成绩。完整可复核结果见 `docs/evidence/final-evaluation.json`，失败案例见同名Markdown。

| 策略 | Recall@3 | MRR@5 | 查询P50秒 | 查询P95秒 |
|---|---:|---:|---:|---:|
| 词级BM25 | 0.750 | 0.737 | 0.0001 | 0.0002 |
| 字符BM25 | 0.889 | 0.847 | 0.0001 | 0.0002 |
| MiniLM Dense | 0.875 | 0.847 | 0.0292 | 0.0333 |
| MiniLM + BM25家族，无翻译 | 0.931 | 0.925 | 0.0334 | 0.0385 |
| BM25家族，无翻译 | 0.819 | 0.808 | 0.0003 | 0.0004 |
| BM25家族 + Qwen翻译 | 0.889 | 0.782 | 0.5752 | 0.8666 |
| MiniLM + BM25家族 + Qwen翻译 | 1.000 | 0.954 | 0.5905 | 1.1835 |
| MiniLM + BM25家族 + Marian单向翻译 | 0.986 | 0.954 | 0.0854 | 0.2332 |
| MiniLM + BM25家族 + DeepSeek翻译 | 1.000 | 0.954 | 0.8567 | 1.2391 |
| Qwen3-Embedding Dense | 1.000 | 0.968 | 0.0622 | 0.0669 |
| Qwen3-Embedding + BM25家族 + Qwen翻译 | 1.000 | 1.000 | 0.5551 | 0.8797 |
| 默认混合检索 + BGE重排前20个Chunk | 1.000 | 0.972 | 0.6886 | 1.0396 |

默认保留 **MiniLM + Qwen查询翻译 + BM25家族 + RRF**。它与更大Embedding组合在当前小集的Recall@3均为1.0，而MiniLM无需另增GPU常驻模型。Qwen3-Embedding将MRR@5从0.954提高到1.0，并在纯Dense组中比MiniLM召回更好；愿意增加约1.15GB GPU模型驻留的用户可以设置 `TRACERAG_EMBEDDING=qwen`。这属于可选取舍，不因小集出现1.0就自动切换默认。

BGE重排在这次实验中将MRR@5从0.954提高到0.972，但Recall@3没有再提升，查询P50从约0.591秒增加到约0.689秒，并需要额外GPU模型驻留。默认保持 `TRACERAG_RERANKER=none`，更重视前排精度时可设置为 `bge`。这次P95受单次运行和查询翻译波动影响，不依据P95的一次下降宣称重排更快。模型单独加载/预热/显存测量记录在 `docs/evidence/model-benchmark.json`。

Qwen和DeepSeek翻译增强的混合检索在此小集上质量相同，Qwen可在下载后离线使用，继续作为默认翻译器。Marian在本组更快，但仅中译英，不满足完整双向需求。纯词法Qwen组的Recall@3提高但MRR下降，说明增加翻译变体能补充召回，也可能改变前排名次，不能只展示有利指标。

下一轮选型应先完成独立人工复核和真实长篇PDF留出集；至少加入不含模型缩写的自然提问、领域专业术语、无答案以及扫描件限制样例。当前资料以英文摘要为主，3篇中文笔记是AI派生材料，跨语言和中文领域覆盖有限。
