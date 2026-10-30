from __future__ import annotations

import argparse
import json
import logging
import glob
from pathlib import Path

from .config import Settings
from .service import RAGService


def main() -> None:
    parser = argparse.ArgumentParser(description="TraceRAG 文档接入、检索问答和本地服务")
    sub = parser.add_subparsers(dest="command",required=True)
    serve = sub.add_parser("serve",help="启动网页和API；模型在第一次查询加载")
    serve.add_argument("--host",default="127.0.0.1")
    serve.add_argument("--port",type=int,default=8000)
    ingest = sub.add_parser("ingest",help="接入一个或多个文档")
    ingest.add_argument("paths",nargs="+",type=Path)
    query = sub.add_parser("query",help="单次检索问答")
    query.add_argument("question")
    query.add_argument("--top-k",type=int,default=5)
    query.add_argument("--language",default="Chinese")
    query.add_argument("--search-only",action="store_true")
    sub.add_parser("chat",help="连续问答，模型和索引仅加载一次")
    sub.add_parser("status",help="查看文档数和配置")
    agent = sub.add_parser("agent",help="受限search/read_chunk工具循环")
    agent.add_argument("question")
    agent.add_argument("--max-steps",type=int,default=6)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.command == "serve":
        import uvicorn
        uvicorn.run("tracerag.api:app",host=args.host,port=args.port,workers=1)
        return
    service = RAGService(Settings.from_env())
    if args.command == "ingest":
        paths=[]
        for supplied in args.paths:
            matches=[Path(p) for p in glob.glob(str(supplied))]
            if not matches:
                raise SystemExit(f"No files match: {supplied}")
            for path in matches:
                paths.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in
                    {".md",".markdown",".txt",".html",".htm",".pdf",".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff"}) if path.is_dir() else [path])
        result = [service.ingest_bytes(p.name,p.read_bytes()) for p in paths]
    elif args.command == "query":
        result = service.query(args.question,args.top_k,args.language,not args.search_only)
    elif args.command == "status":
        result = service.status()
    elif args.command == "agent":
        result = service.agent(args.question,max_steps=args.max_steps)
    else:
        print("输入问题开始查询；输入 exit 退出。首次查询加载模型，后续复用。")
        while True:
            try:
                question = input("问题> ").strip()
            except (EOFError,KeyboardInterrupt):
                break
            if question.lower() in {"exit","quit"}:
                break
            if not question:
                continue
            try:
                print(json.dumps(service.query(question),ensure_ascii=False,indent=2))
            except (ValueError,RuntimeError) as exc:
                print(str(exc))
        return
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
