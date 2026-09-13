# 接口定义

TraceRAG 将业务编排与模型供应商、检索算法分离。以下签名是项目合约；Day 1 只实现了 LLM 边界，其余为计划接口。

## 已实现：模型客户端

```python
class LLMClient(Protocol):
    def complete(self, prompt: str) -> str:
        ...
```

输入是已构造好的 Prompt，输出是模型文本。Mock 客户端无需网络即可满足此合约；未来 MaaS 客户端也必须提供相同方法。

## 计划接口

```python
class DocumentParser(Protocol):
    def parse(self, path: Path) -> Sequence[Document]:
        ...

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
