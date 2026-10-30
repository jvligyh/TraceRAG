# Day 4：BM25 词法检索

## 今日目标

- [x] 实现中英文确定性分词：中文使用 jieba 精确模式，英文保留技术词。
- [x] 关闭 jieba HMM，并用 RAG 领域短语扩展兼顾宽召回和精确匹配。
- [x] 实现独立的中文字符 2/3-gram tokenizer，为未知专业词提供词典外召回。
- [x] 使用双向查询翻译让中文问题召回英文文献、英文问题召回中文文献。
- [x] 将词级、字符级和翻译查询压成一张 BM25 家族排名，避免同语言结果重复投票。
- [x] 建立词频、文档频率和倒排表。
- [x] 实现带词频饱和与文档长度归一化的 BM25 评分。
- [x] 复用 `Retriever` 返回结构，保留 Chunk、score、rank 和来源。
- [x] 增加英文、中文、同分排序和异常路径测试。
- [x] 使用真实演示文档跑通一次 BM25 查询。

## 当前 Baseline

- `k1=1.5`：控制同一个词重复出现时，词频收益多快饱和。
- `b=0.75`：控制文档长度归一化强度。
- 英文与技术词：NFKC 归一化后转小写，保留 `bm25`、`c++`、`recall@k` 等词。
- 中文：使用 `jieba==0.42.1` 精确模式进行词级切分，并关闭 HMM 保持结果可复现。例如“检索评估”变为“检索、评估”，不会再制造无意义的“索评”。
- 领域短语扩展：保留“向量、检索”等基础词，同时额外加入“向量检索”完整短语。这样查询“检索”仍能召回含“向量检索”的文档，而完整短语查询又能得到额外匹配。
- 混合文本：仍通过 NFKC 统一全角字符、英文转小写，并保留 `bm25`、`c++`、`recall@k` 等技术词。
- 字符通道：对连续中文分别生成 2-gram 和 3-gram，例如“钙钛矿”得到“钙钛、钛矿、钙钛矿”；标点会截断窗口，不生成跨句片段。
- 两个通道分别建立 BM25 索引，不把词和字符片段塞进同一个原始分数。
- 遇到包含中文的查询时，`LLMQueryTranslator` 生成英文检索查询；英文查询复用词级 BM25 索引，从而召回英文文献。
- 词级、字符级、英文翻译三张词法榜对每个 Chunk 只取最佳倒数名次，压成一张 BM25 家族榜。Day 5 只让 BM25 家族和 Dense 各投一票。
- 查询复杂度：只遍历查询词对应的倒排表，而不是扫描每个 Chunk 的全文。
- 实测证据：`docs/evidence/day4-bm25.md`

## 当前边界

- jieba 依赖固定词典，遇到新产品名、人名和垂直领域术语时仍可能切错，需要通过真实查询集继续补充领域短语。
- 领域短语表目前由人工维护，后续应根据失败案例更新，而不是无限堆词。
- 字符 n-gram 会生成一些没有独立含义的片段，也会增加索引体积，因此它是补充召回通道，不替代词级 BM25。
- 分词优化只能改善中文词法匹配，不能让 BM25 理解同义改写或自动跨语言；这部分仍由 Dense Retriever 补足。
- 跨语言能力来自查询翻译，不是 BM25 公式本身突然理解了中英文；翻译失败、术语漂移和额外 API 延迟都必须单独评测。
- Day 7 将在固定评测集上比较召回率，而不是仅凭几个分词示例判断优化是否有效。

## 你需要亲手完成

- [x] 用自己的话解释 TF、DF 和 IDF。
- [x] 解释为什么高频但到处出现的词区分度低。
- [x] 解释 `k1` 和 `b` 分别控制什么。
- [x] 逐行阅读 `bm25.py` 和 `test_bm25.py`。
- [x] 运行真实脚本，检查 top-1 是否与英文精确词相关。
- [x] 分别运行英文查询和中文查询，观察 BM25 不能自动跨语言匹配。
- [x] 将 `--top-k` 从 1 改为 3，观察只返回有词法命中的 Chunk。
- [x] 亲手增加一项 BM25 边界测试。

## 运行

```powershell
python scripts/day4_bm25.py data/demo/rag_evaluation.md data/demo/traceability.md `
  --query "source identifiers citations" --top-k 3 --max-chars 180 --overlap-chars 30
```

字符 2/3-gram BM25：

```powershell
python scripts/day4_bm25.py <中文文献路径> `
  --query "钙钛矿结构" --analyzer char-ngram --top-k 3
```

跨语言 BM25：

```powershell
python scripts/day4_crosslingual_bm25.py `
  data/demo/rag_evaluation.md data/demo/traceability.md `
  --query "为什么要分别评估检索和生成？" --top-k 3
```

全部测试：

```powershell
python -m unittest discover -s tests -v
```
