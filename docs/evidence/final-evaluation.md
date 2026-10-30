# 检索消融实测

AI 起草、证据字符串核对过的种子评测集，尚未独立人工复核；这些分数不代表真实用户或长篇论文效果。

语料 28 篇，115 个 Chunk，问题 40 条。固定 corpus SHA256：`061498764b72737520f21851a8c8a2b502977e30c91eebb51dbe190f6fadc0ba`。

| 策略 | 状态 | Recall@1 | Recall@3 | Recall@5 | MRR@5 | P50秒 | P95秒 |
|---|---|---:|---:|---:|---:|---:|---:|
| bm25 | completed | 0.556 | 0.750 | 0.931 | 0.737 | 0.0001 | 0.0002 |
| char | completed | 0.694 | 0.889 | 0.917 | 0.847 | 0.0001 | 0.0002 |
| dense | completed | 0.708 | 0.875 | 0.889 | 0.847 | 0.0292 | 0.0333 |
| hybrid-none | completed | 0.806 | 0.931 | 0.986 | 0.925 | 0.0334 | 0.0385 |
| lexical-none | completed | 0.639 | 0.819 | 0.931 | 0.808 | 0.0003 | 0.0004 |
| lexical-qwen | completed | 0.556 | 0.889 | 1.000 | 0.782 | 0.5752 | 0.8666 |
| hybrid-qwen | completed | 0.833 | 1.000 | 1.000 | 0.954 | 0.5905 | 1.1835 |
| hybrid-marian | completed | 0.833 | 0.986 | 0.986 | 0.954 | 0.0854 | 0.2332 |
| hybrid-deepseek | completed | 0.833 | 1.000 | 1.000 | 0.954 | 0.8567 | 1.2391 |
| dense-qwen | completed | 0.861 | 1.000 | 1.000 | 0.968 | 0.0622 | 0.0669 |
| hybrid-qwen-embedding | completed | 0.917 | 1.000 | 1.000 | 1.000 | 0.5551 | 0.8797 |
| hybrid-rerank | completed | 0.861 | 1.000 | 1.000 | 0.972 | 0.6886 | 1.0396 |

MRR 只看前5个去重文档。无答案题单独报告空检索比例，不计入 Recall 分母；Dense 始终有最近邻，不能把返回文档当作可回答的证据。

完整 JSON 包含每题相关ID/返回ID/失败清单、分组结果和模型环境。引用ID有效率只校验ID是否合法，不替代事实支持度与答案正确性。

尚未执行的模型组合没有分数；更大模型不预设优于基线。

## Recall@3 未命中案例

| 策略 | 题目ID | 问题 |
|---|---|---|
| bm25 | q02 | 原始 RAG 的非参数记忆存储了什么语料？ |
| bm25 | q10 | BEIR 中平均零样本效果最好的重排和后期交互方法付出了什么代价？ |
| bm25 | q16 | C-MTEB 的中文文本向量基准覆盖多少任务和数据集？ |
| bm25 | q18 | Contriever 可以从阿拉伯语查询检索英文文献吗？ |
| bm25 | q20 | HyDE 的假设文档是否可能包含虚假的细节？ |
| bm25 | q26 | Self-RAG 是否为每个问题固定检索相同数量的文档？ |
| bm25 | q30 | BRIGHT 中的复杂查询为什么不能只做关键词或表面语义匹配？ |
| char | q02 | 原始 RAG 的非参数记忆存储了什么语料？ |
| char | q18 | Contriever 可以从阿拉伯语查询检索英文文献吗？ |
| dense | q04 | DPR 相比 Lucene-BM25 的 top-20 检索准确率提升了多少个百分点？ |
| dense | q10 | BEIR 中平均零样本效果最好的重排和后期交互方法付出了什么代价？ |
| dense | q23 | Which passage re-ranking paper reports a 27 percent relative improvement in MRR@10? |
| hybrid-none | q02 | 原始 RAG 的非参数记忆存储了什么语料？ |
| hybrid-none | q18 | Contriever 可以从阿拉伯语查询检索英文文献吗？ |
| lexical-none | q02 | 原始 RAG 的非参数记忆存储了什么语料？ |
| lexical-none | q10 | BEIR 中平均零样本效果最好的重排和后期交互方法付出了什么代价？ |
| lexical-none | q16 | C-MTEB 的中文文本向量基准覆盖多少任务和数据集？ |
| lexical-none | q18 | Contriever 可以从阿拉伯语查询检索英文文献吗？ |
| lexical-none | q26 | Self-RAG 是否为每个问题固定检索相同数量的文档？ |
| lexical-qwen | q02 | 原始 RAG 的非参数记忆存储了什么语料？ |
| lexical-qwen | q10 | BEIR 中平均零样本效果最好的重排和后期交互方法付出了什么代价？ |
| lexical-qwen | q16 | C-MTEB 的中文文本向量基准覆盖多少任务和数据集？ |
| lexical-qwen | q26 | Self-RAG 是否为每个问题固定检索相同数量的文档？ |
