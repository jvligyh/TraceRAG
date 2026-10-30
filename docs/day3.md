# Day 3：稠密检索 Baseline

## 今日目标

- [x] 定义查询/文档分离的 `EmbeddingModel` 接口。
- [x] 接入本地多语言 FastEmbed 模型。
- [x] 实现余弦相似度。
- [x] 实现内存穷举 top-k 稠密检索。
- [x] 返回 Chunk、score 和 rank，不丢失来源元数据。
- [x] 增加不依赖真实模型下载的确定性单元测试。
- [x] 使用真实模型跑通一次中英文交叉检索。

## 当前 Baseline

- 模型：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- 向量维度：384
- 模型覆盖：多语言，适合当前英文公开资料和中文查询/资料
- 索引方式：所有 Chunk 向量暂存在内存
- 搜索方式：查询与所有 Chunk 逐一计算余弦相似度，再排序取 top-k
- 复杂度：每次查询约为 `O(N × D)`，其中 N 是 Chunk 数，D 是向量维度
- 实测证据：`docs/evidence/day3-dense-retrieval.md`

小规模 Baseline 暂不接入向量数据库。先验证检索质量，再在数据量确实需要时引入 ANN 索引，避免把基础设施复杂度误写成算法效果。

## Day 7 后续实验

当前模型只作为 Baseline 保留。Day 6 建立人工标注集后，Day 7 将它与 Qwen3-Embedding-0.6B、BM25 和 RRF 组合进行受控消融，再根据 Recall@k、MRR、延迟和资源占用决定是否升级。实验设计见 `docs/day7-ablation-plan.md`。

## 你需要亲手完成

- [x] 用自己的话解释 Embedding 是什么。
- [x] 手算 `(1, 0)` 与 `(1, 1)` 的余弦相似度。
- [x] 解释为什么查询使用 `query_embed`，文档使用 `passage_embed`。
- [x] 逐行阅读 `embeddings.py`、`retrieval.py` 和 `test_retrieval.py`。
- [x] 运行真实模型脚本，并检查 top-1 是否与问题相关。
- [x] 将 `--top-k` 从 1 改为 3，观察结果数量和排序。
- [x] 亲手增加一项检索边界测试。

## 运行

首次运行会下载约 220 MB 的本地模型：

```powershell
python scripts/day3_dense.py data/demo/rag_evaluation.md data/demo/traceability.md `
  --query "为什么要分别评估检索和生成？" --top-k 3 --max-chars 180 --overlap-chars 30
```

全部测试：

```powershell
python -m unittest discover -s tests -v
```
