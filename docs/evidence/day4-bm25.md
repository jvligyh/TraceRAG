# Day 4 BM25 词法检索证据

日期：2026-09-17

## 配置

- 语料：`rag_evaluation.md`、`traceability.md`
- 文档数：2
- Chunk 数：7
- `max_chars=180`
- `overlap_chars=30`
- `k1=1.5`
- `b=0.75`

## 英文精确词查询

查询：`source identifiers citations`

- 返回 2 个有词面命中的 Chunk，而不是用零分结果凑满 top-k。
- top-1 来源：`traceability.md`
- top-1 score：3.081863
- top-1 文本同时包含 `source` 和 `identifiers`。
- top-2 文本包含 `identifiers` 和 `citations`。
- 建索引约 0.000425 秒，查询约 0.000033 秒。

## 跨语言边界

原始 BM25 对相同的两篇英文文档查询中文「为什么要分别评估检索和生成？」时返回空结果，因为中文词与英文 token 没有词面重合。加入查询翻译后，DeepSeek 实际返回 `Why separately evaluate retrieval and generation?`，翻译增强 BM25 的 top-1 正确命中 `rag_evaluation.md` 中包含 `evaluate retrieval and generation separately` 的 Chunk。

本次冷启动实测中，DeepSeek 翻译约 1.63 秒，翻译后的本地检索约 0.68 秒；后者包含 jieba 首次加载约 0.68 秒，不能当作预热后的稳定延迟。相同查询的翻译结果使用进程内 LRU 缓存，后续消融实验仍需分别记录冷启动和预热延迟。

词级原查询、字符原查询和英文翻译查询不会作为三票直接进入全局融合。`CrossLingualBM25Retriever` 对同一 Chunk 只保留最佳倒数名次，先压成一张 BM25 家族榜；Day 5 再与 Dense 榜进行 RRF。

## 字符 n-gram 通道

- 中文连续文本默认同时生成 2-gram 和 3-gram，标点作为边界。
- 合成跨领域测试包含“钙钛矿”和“石墨烯”两个 Chunk；查询“钙钛矿结构”时，字符 BM25 将钙钛矿 Chunk 排在 top-1。
- 英文与技术词仍按完整 token 保留。使用 `--analyzer char-ngram` 重跑英文查询时，top-1 仍为 `traceability.md`，score 为 3.081863。
- 字符索引与 jieba 词级索引相互独立，当前不直接相加分数；计划由 Day 5 RRF 融合排名。

## 验证

- 当前测试覆盖查询翻译缓存、空翻译、中文查询召回英文文献、英文查询召回中文文献，以及多词法变体去重。
- 覆盖英文技术词、中文词级切分、领域短语扩展、字符 2/3-gram、标点边界、词典外专业词召回、相关性排序、同分排序、未知词和非法参数。
