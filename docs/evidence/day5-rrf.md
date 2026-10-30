# Day 5 RRF 融合证据

日期：2026-09-18

## 配置

- 语料：`rag_evaluation.md`、`traceability.md`
- 文档数：2
- Chunk 数：7
- 查询：`为什么要分别评估检索和生成？`
- 查询翻译：本地 Qwen3-0.6B、MarianMT 与 DeepSeek 对照
- Dense：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `candidate_k=5`
- `fusion_k=60`

## 对照结果

关闭查询翻译时，BM25 家族无法用中文 token 命中英文语料，最终结果只有 Dense 排名贡献。Dense top-1 是讨论 `retrieval and generation separately` 的 Chunk。

## 完整融合结果

本地 Qwen3-0.6B 返回：`Why separate evaluation of retrieval and generation?`

Qwen3 反向验证：`Why should retrieval and generation be evaluated separately?` 返回 `检索与生成应分开评估`。因此当前默认翻译器覆盖中文问英文文献和英文问中文文献两种方向。

本地 MarianMT 返回：`Why do we evaluate retrieval and generation separately?`

DeepSeek 返回：`Why evaluate retrieval and generation separately?`

Qwen3、Marian 和 DeepSeek 的中译英结果都让 BM25 家族召回相同的英文相关 Chunk，本次最终 top-3 排名一致。

## 本地翻译失败案例

Marian 在通用 RAG 查询上表现正常，但跨领域压力测试暴露了明显术语错误：

- “钙钛矿太阳能电池”被错误翻译为与 `thorium-mining solar cells` 相关的表达；
- “单细胞转录组”被错误翻译为 `single cell transcoder group`。

Qwen3 对相同压力问题给出了 `perovskite solar cells`、`single-cell transcriptomic sequencing` 和 `batch effects` 等正确术语，当前样例优于 Marian；但4条人工样例不足以证明整体质量。Day 7 仍需在固定专业术语子集上比较 Qwen3、Marian、DeepSeek 和不翻译四组的 Recall@k。

| 最终名次 | BM25 家族名次 | Dense 名次 | RRF score | 结果摘要 |
|---:|---:|---:|---:|---|
| 1 | 2 | 1 | 0.03252247 | `retrieval and generation separately` |
| 2 | 1 | 2 | 0.03252247 | `evaluate retrieval and generation separately` |
| 3 | 5 | 3 | 0.03125763 | 分别评估能定位失败组件 |

前两项的 RRF 分数相同，因为它们的名次组合分别是 `(2,1)` 与 `(1,2)`。实现继续比较最佳名次；仍相同时按稳定 `chunk_id` 排序，确保重复运行结果一致。

## 延迟样本

- 建索引：约1.54秒；
- Qwen3 首次下载并加载：约300秒（一次性网络成本，不作为日常延迟）；
- Qwen3 缓存加载：约5.25～6.67秒；
- Qwen3 CPU 中译英：约2.25～3.05秒；
- Qwen3 CPU 英译中：约1.55～1.75秒；
- PyTorch CUDA：`2.14.0+cu130`，RTX 4060 Laptop GPU，compute capability 8.9；
- Qwen3 CUDA 中译英：约1.28～1.36秒；
- Qwen3 CUDA 英译中：约0.29秒；
- Qwen3 CUDA 峰值显存：allocated 约1155 MB，reserved 约1172 MB；
- Marian 缓存加载：约4.59秒；
- Marian CPU 中译英：约0.46秒；
- DeepSeek 翻译：约1.85秒；
- BM25 家族查询：约0.56秒，其中包含 jieba 冷启动；
- Dense 查询：约0.005秒；
- RRF 融合：约0.00004秒。

这些数字来自一次小语料运行，不代表稳定性能。正式报告需要预热并统计多次运行的 P50/P95。

FastEmbed 在本次运行中提示该模型当前使用 mean pooling，而旧配置使用 CLS pooling。功能链路仍然通过，但这说明只固定包版本还不足以完全固定远端模型配置；Day 7 实验前需要冻结模型文件与 pooling 配置。

## 验证范围

- 仓库全部测试通过；
- 已知排名下的 RRF 公式；
- 忽略不可比较的原始 retrieval score；
- 按 `chunk_id` 去重；
- 同一榜内重复 Chunk 拒绝；
- 相同 ID 对应冲突数据拒绝；
- 稳定平局顺序；
- 空榜与非法参数；
- Wrapper 向子检索器传递查询和 `candidate_k`。
- 本地模型优先解析固定 revision 的本地 snapshot，仅在缓存缺失或不完整时回退下载；已在禁止联网的运行环境中验证 Qwen 可加载与双向翻译。
