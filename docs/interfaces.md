# 接口定义

TraceRAG 将业务编排与模型供应商、文档格式和检索算法分离。LLM 与文档解析边界已实现，检索与评测仍是计划接口。

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

## 计划接口

```python
class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> Sequence[RetrievedChunk]:
        ...

class Reranker(Protocol):
    def rerank(self, query: str, chunks: Sequence[RetrievedChunk]) -> Sequence[RetrievedChunk]:
        ...

class Evaluator(Protocol):
    def evaluate(self, cases: Sequence[EvaluationCase]) -> EvaluationReport:
        ...
```

「计划接口」只是设计文档，不能写进简历的已实现成果。只有配套代码和测试完成后，才能将状态改为「已实现」。
