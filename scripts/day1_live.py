from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracerag import Document, TraceRAGPipeline  # noqa: E402
from tracerag.providers import DeepSeekLLMClient  # noqa: E402


def main() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        api_key = getpass.getpass("DeepSeek API key: ").strip()
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is required")

    client = DeepSeekLLMClient(
        api_key,
        api_base=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
    )
    pipeline = TraceRAGPipeline(client)
    documents = [
        Document(
            document_id="day1-grounding",
            title="Day 1 验收信息",
            text="TraceRAG 的 Day 1 目标是建立供应商无关的模型接口，并完成一次真实模型调用。",
        )
    ]
    response = pipeline.answer("TraceRAG Day 1 要完成哪两个与模型有关的目标？", documents)

    print(f"provider=deepseek model={client.model}")
    print(f"answer={response.answer}")
    print(f"citations={', '.join(response.citations)}")


if __name__ == "__main__":
    main()
