from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracerag import Document, TraceRAGPipeline  # noqa: E402
from tracerag.providers import MockLLMClient  # noqa: E402


def load_demo_documents() -> list[Document]:
    documents: list[Document] = []
    for path in sorted((ROOT / "data" / "demo").glob("*.md")):
        documents.append(
            Document(document_id=path.stem, title=path.stem.replace("_", " "), text=path.read_text(encoding="utf-8"))
        )
    return documents


def main() -> None:
    pipeline = TraceRAGPipeline(MockLLMClient())
    response = pipeline.answer("Why does a RAG system need offline evaluation?", load_demo_documents())
    print(response.answer)
    print("citations:", ", ".join(response.citations))


if __name__ == "__main__":
    main()

