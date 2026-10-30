"""下载公开论文摘要与官方文档；保存来源与文件摘要，不把摘要冒称全文。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


class ArticleText(HTMLParser):
    def __init__(self, abstract: bool) -> None:
        super().__init__(convert_charrefs=True)
        self.abstract = abstract
        self.depth = 0
        self.ignored = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag in {"script", "style", "nav"}:
            self.ignored += 1
        if self.depth:
            if tag not in {"br", "img", "hr", "input", "meta", "link", "source", "wbr"}:
                self.depth += 1
        elif (self.abstract and tag == "blockquote" and "abstract" in attributes.get("class", "")) or (
            not self.abstract and tag in {"main", "article"}
        ):
            self.depth = 1
        if tag in {"p", "h1", "h2", "h3", "li", "pre", "br"} and self.depth:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav"}:
            self.ignored = max(0, self.ignored - 1)
        if self.depth and tag not in {"br", "img", "hr", "input", "meta", "link", "source", "wbr"}:
            self.depth -= 1

    def handle_data(self, data: str) -> None:
        if self.depth and not self.ignored:
            self.parts.append(data)

    def text(self) -> str:
        text = " ".join(self.parts)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s*\n\s*", "\n", text).strip()
        return re.sub(r"^Abstract:\s*", "", text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/corpus")
    parser.add_argument("--refresh", action="store_true", help="重新下载已有文件")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [json.loads(line) for line in (ROOT / "data/sources/public_corpus.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    old_manifest = args.output / "manifest.json"
    previous = {item["source_id"]: item for item in json.loads(old_manifest.read_text(encoding="utf-8")).get("documents", [])} if old_manifest.exists() else {}
    documents = []
    failures = []
    for item in sources:
        path = args.output / f"{item['id']}.md"
        if path.exists() and not args.refresh and item["id"] in previous:
            documents.append(previous[item["id"]])
            continue
        try:
            request = Request(item["url"], headers={"User-Agent": "TraceRAG-educational-corpus/0.1"})
            with urlopen(request, timeout=40) as response:
                raw = response.read(5_000_000).decode("utf-8", errors="replace")
                resolved_url = response.url
            article = ArticleText(abstract=item["kind"] == "paper")
            article.feed(raw)
            text = article.text()
            if len(text) < 100:
                raise ValueError("main text / abstract could not be extracted")
            scope = "abstract" if item["kind"] == "paper" else "official-documentation-excerpt"
            if scope != "abstract":
                text = text[:18000]
            content = f"# {item['title']}\n\nSource: {item['url']}\nScope: {scope}; not a full-paper archive.\n\n{text}\n"
            path.write_text(content, encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            documents.append({"source_id": item["id"], "title": item["title"], "url": item["url"], "resolved_url": resolved_url, "scope": scope, "file": path.name, "sha256": digest, "document_id": f"md-{digest[:16]}", "fetched_at": datetime.now(timezone.utc).isoformat(), "characters": len(content)})
            print(f"OK {item['id']}: {len(content)} characters", flush=True)
        except Exception as exc:
            failures.append({"source_id": item["id"], "error": type(exc).__name__})
            print(f"FAILED {item['id']}: {type(exc).__name__}", flush=True)
        time.sleep(3 if item["kind"] == "paper" else 0.5)
    derived = []
    for path in sorted(args.output.glob("zh-*.md")):
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        url_match = re.search(r"^Source: (.+)$", text, flags=re.MULTILINE)
        digest = hashlib.sha256(raw).hexdigest()
        derived.append({"source_id": path.stem, "file": path.name, "url": url_match.group(1) if url_match else None, "scope": "AI-authored Chinese reading note; pending human review; not an official translation", "sha256": digest, "document_id": f"md-{digest[:16]}"})
    manifest = {"description": "Real public paper abstracts and official documentation excerpts, not full papers.", "documents": documents, "derived_documents": derived, "failures": failures}
    old_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(documents)}/{len(sources)} documents; {len(failures)} failures")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
