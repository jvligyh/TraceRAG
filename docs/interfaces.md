# 接口定义

TraceRAG 将业务编排与模型供应商、文档格式和检索算法分离。0.2.0已实现文档解析、Embedding、BM25/Dense、RRF、可选重排、端到端生成引用、受限工具循环、评测和HTTP服务。原TraceRAGPipeline保留教学用途，应用入口使用RAGService。

## 应用入口

`RAGService(Settings.from_env())`提供`ingest_bytes(filename,content)`、`list_documents()`、`delete_document(id)`、`query(question,top_k,answer_language,generate)`和`agent(question,max_steps,answer_language)`。query返回answer/status/citations/results/timings；citations包含经过供给证据ID校验的chunk元数据。它不是由传入Document自动生成的全量引用清单。

`CrossEncoderReranker.rerank(query,chunks,top_k)`接受RRF候选榜，返回重新排序的RetrievedChunk；其score为模型logit。`TransformerEmbeddingModel`满足EmbeddingModel协议，使用Qwen3最后有效token池化与归一化。evaluation.py独立计算文档级Recall/MRR/延迟，答案事实正确性另需人工审核。

## 已实现：模型客户端

```python
class LLMClient(Protocol):
    def complete(self, prompt: str) -> str:
        ...
```

输入是已构造好的 Prompt，输出是模型文本。`MockLLMClient` 与 `DeepSeekLLMClient` 都已满足该合约。

## 已实现：文档解析器

```python
class DocumentParser(Protocol):
    def parse(self, path: Path) -> tuple[Document, ...]:
        ...
```

`MarkdownParser` 与 `PDFParser` 都满足该接口。

## 已实现：Embedding 与稠密检索

```python
class EmbeddingModel(Protocol):
    def embed_passages(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        ...

    def embed_query(self, text: str) -> Embedding:
        ...

class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> tuple[RetrievedChunk, ...]:
        ...
```

`FastEmbedEmbeddingModel` 使用本地多语言模型；`DenseRetriever` 在内存中计算余弦相似度并返回 top-k。真实模型与单元测试假模型满足同一接口。

`BM25Retriever` 也满足同一个 `Retriever` 接口。默认 tokenizer 提供 jieba 词级检索；传入 `tokenize_char_ngrams` 时，同一个实现建立中文字符 2/3-gram 索引。两个实例先组成 BM25 家族榜，再与 Dense 交给融合层处理，三种原始分数不会直接相加。

`CrossLingualBM25Retriever` 同样满足 `Retriever`。它通过 `QueryTranslator` 双向扩展查询：中文问题补充英文译文，英文问题补充中文译文；原查询和译文都进入词级与字符级索引。各词法变体先按“每个 Chunk 的最佳名次”压成一张 BM25 家族榜，避免相关通道在最终融合时重复投票。`LLMQueryTranslator` 依赖已有的 `LLMClient`，单元测试可以换成静态翻译器而不访问网络。

`QwenQueryTranslator` 是默认本地实现，使用固定 commit 的 `Qwen/Qwen3-0.6B` 执行中英双向查询翻译。它先把固定 revision 解析成本地 snapshot 路径，避免 Transformers 在离线加载 tokenizer 时仍访问远端 API。`MarianQueryTranslator` 保留为更轻量的中译英基线，但 `Helsinki-NLP/opus-mt-zh-en` 不支持英译中。

查询翻译只决定“用哪些词去找文献”，不决定最终回答语言。`TraceRAGPipeline.answer(..., answer_language="Chinese")` 会单独要求生成模型用中文回答，因此可以实现“英文提问、检索中英文文献、中文作答”，而不会把查询译文误当成答案。

## 已实现：RRF 融合

`ReciprocalRankFusionRetriever` 接收至少两个满足 `Retriever` 的对象，对每个子检索器请求候选榜，再通过 `reciprocal_rank_fusion()` 仅按名次求和。它不直接相加 BM25、余弦相似度或其他不可比较的原始 score，并使用 `chunk_id` 去重。

## 计划接口

```python
class Reranker(Protocol):
    def rerank(self, query: str, chunks: Sequence[RetrievedChunk]) -> Sequence[RetrievedChunk]:
        ...

class Evaluator(Protocol):
    def evaluate(self, cases: Sequence[EvaluationCase]) -> EvaluationReport:
        ...
```

「计划接口」只是设计文档，不能写进简历的已实现成果。只有配套代码和测试完成后，才能将状态改为「已实现」。
