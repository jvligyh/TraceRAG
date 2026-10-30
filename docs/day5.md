# Day 5：RRF 融合与去重

## 今日目标

- [x] 实现不依赖原始 score 的 Reciprocal Rank Fusion。
- [x] 使用 `chunk_id` 合并多个检索器返回的同一 Chunk。
- [x] 同一个检索器不得为一个 Chunk 重复投票。
- [x] 相同 `chunk_id` 对应不同内容时立即报错。
- [x] 使用融合分数、最佳名次和 `chunk_id` 建立确定性排序。
- [x] 实现满足统一 `Retriever` 接口的 `ReciprocalRankFusionRetriever`。
- [x] 将 BM25 家族作为一票、Dense 作为一票进行融合。
- [x] 使用中文问题、英文文献和真实 DeepSeek 翻译跑通完整链路。
- [x] 接入固定 revision 的本地 Qwen3-0.6B 中英双向翻译模型，并优先读取离线缓存。
- [x] 保留 MarianMT 作为更轻量、但仅支持中译英的对照基线。

## RRF 原理

某个 Chunk 的融合分数为：

```text
RRF(chunk) = Σ 1 / (fusion_k + rank_i)
```

当前 `fusion_k=60`。RRF 只读取各检索器给出的名次，不读取 BM25 分数、余弦相似度或 BM25 家族分数，因为这些 score 的含义和范围不同，不能直接相加。

例如一个 Chunk 在 BM25 家族排第2、Dense 排第1：

```text
1 / (60 + 2) + 1 / (60 + 1)
= 0.03252247
```

## 两级公平融合

```text
原查询词级 BM25 ─┐
原查询字符 BM25 ─┤
译文词级 BM25 ───┼─ 最佳词法名次去重 ─> 一张 BM25 家族榜 ─┐
译文字符 BM25 ───┘                                         ├─ RRF ─> 最终排名
Dense ──────────────────────────────────────────────────────┘
```

第一层对相关词法通道只取最佳名次，防止中文文献因为同时命中词级与字符级而重复投票。第二层标准 RRF 对 BM25 家族榜和 Dense 榜求和，因此两个检索家族权重对等。

## 三个参数不要混淆

- `top_k`：最终返回多少个 Chunk。
- `candidate_k`：每个检索器先提供多少个候选，通常应大于 `top_k`。
- `fusion_k`：RRF 公式中的平滑常数，不是结果数量。它越大，不同名次之间的分差越小。

## 真实结果

问题：`为什么要分别评估检索和生成？`

- 本地 Qwen3 翻译：`Why separate evaluation of retrieval and generation?`
- Qwen3 英译中验证：`Why should retrieval and generation be evaluated separately?` → `检索与生成应分开评估`
- 本地 MarianMT 翻译：`Why do we evaluate retrieval and generation separately?`
- DeepSeek 翻译：`Why evaluate retrieval and generation separately?`
- 最终 top-1：`rag_evaluation.md` 中讨论 `retrieval and generation separately` 的 Chunk。
- top-1 的 BM25 家族名次为2，Dense 名次为1，RRF score 为 `0.03252247`。
- 另一个 Chunk 的 BM25 家族名次为1、Dense 名次为2，得到相同融合分数；当前使用稳定 `chunk_id` 作为最终平局规则。
- Qwen3 缓存加载约5.25～8.36秒。CPU 环境中译英约2.25～3.05秒、英译中约1.55～1.75秒；换成 `torch 2.14.0+cu130` 后，RTX 4060 Laptop GPU 中译英约1.28～1.36秒、英译中约0.29秒，峰值显存约1.17 GB。DeepSeek 翻译约1.85秒，Marian CPU 中译英约0.46秒。本地 Dense 查询约0.007秒，RRF 本身约0.00005秒。该数据只代表当前机器与少量单次运行。
- 跨领域压力测试发现 Marian 会把“钙钛矿”和“单细胞转录组”译错，因此它目前只是离线 Baseline，不是已经验证过的最终翻译器；系统仍保留 DeepSeek 选项，并由 Dense 兜底语义召回。
- FastEmbed 本次发出了模型 pooling 配置发生变化的警告，因此这些结果只能作为功能验收，Day 7 正式实验前必须冻结并记录模型配置。

## 你需要亲手完成

- [x] 手算 `fusion_k=60`、名次分别为2和1时的 RRF 分数。
- [x] 用自己的话解释为什么不能直接相加 BM25 score 和余弦相似度。
- [x] 解释 `top_k`、`candidate_k`、`fusion_k` 的区别。
- [x] 阅读 `fusion.py` 和 `test_fusion.py`。
- [x] 分别使用 `qwen`、`marian`、`deepseek`、`none` 运行脚本，观察译文、延迟和 BM25 家族名次变化。
- [x] 用中文问题和英文问题各运行一次，确认译文方向相反。
- [x] 将 `--top-k` 从1改为3，检查最终结果和组件名次。
- [x] 亲手增加一项 RRF 边界测试。

## 运行

完整跨语言融合：

```powershell
python scripts/day5_rrf.py data/demo/rag_evaluation.md data/demo/traceability.md `
  --query "为什么要分别评估检索和生成？" `
  --top-k 3 --candidate-k 5 --translator qwen
```

关闭翻译作为对照：

```powershell
python scripts/day5_rrf.py data/demo/rag_evaluation.md data/demo/traceability.md `
  --query "为什么要分别评估检索和生成？" `
  --top-k 3 --candidate-k 5 --translator none
```

在线 DeepSeek 对照：

```powershell
python scripts/day5_rrf.py data/demo/rag_evaluation.md data/demo/traceability.md `
  --query "为什么要分别评估检索和生成？" `
  --top-k 3 --candidate-k 5 --translator deepseek
```

全部测试：

```powershell
python -m unittest discover -s tests -v
```
