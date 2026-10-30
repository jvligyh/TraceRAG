"""HTTP boundary: bounded uploads, validated requests, and a local browser client."""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4
from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .errors import DocumentInputError


LOGGER = logging.getLogger("tracerag.http")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf", ".html", ".htm"} | IMAGE_SUFFIXES
IMAGE_MEDIA_TYPES = {
    "PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp",
    "BMP": "image/bmp", "TIFF": "image/tiff",
}


class BodyLimitMiddleware:
    """Bound the actual streamed body, including requests without Content-Length."""

    def __init__(self, app: Any, max_upload_bytes: int = MAX_UPLOAD_BYTES) -> None:
        self.app = app
        self.max_upload_bytes = max_upload_bytes

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self.max_upload_bytes + 1024 * 1024 if scope["path"] == "/documents" else 64 * 1024
        headers = dict(scope.get("headers", []))
        try:
            content_length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            await JSONResponse({"detail": "无效的 Content-Length"}, status_code=400)(scope, receive, send)
            return
        if content_length > limit:
            await JSONResponse({"detail": "请求内容超过大小限制"}, status_code=413)(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > limit:
                await JSONResponse({"detail": "请求内容超过大小限制"}, status_code=413)(scope, receive, send)
                return
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay() -> dict:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


class HistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)
    answer_language: Literal["Chinese", "English", "中文", "英文"] = "Chinese"
    generate: bool = True
    document_ids: list[str] | None = Field(default=None, max_length=100)
    history: list[HistoryMessage] | None = Field(default=None, max_length=12)

    @model_validator(mode="after")
    def bounded_history(self):
        if sum(len(message.content) for message in self.history or ()) > 16000:
            raise ValueError("conversation history exceeds 16000 characters")
        return self


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=4000)
    max_steps: int = Field(default=4, ge=1, le=8)
    answer_language: Literal["Chinese", "English", "中文", "英文"] = "Chinese"
    document_ids: list[str] | None = Field(default=None, max_length=100)


def create_app(service: Any = None, *, max_upload_bytes: int = MAX_UPLOAD_BYTES) -> FastAPI:
    if max_upload_bytes < 1:
        raise ValueError("max_upload_bytes must be positive")
    app = FastAPI(title="TraceRAG", version="0.4.0", description="可追溯的中英文文档问答与本地OCR")
    app.state.service = service
    app.state.service_lock = Lock()
    app.add_middleware(BodyLimitMiddleware, max_upload_bytes=max_upload_bytes)

    def get_service() -> Any:
        # Model loading belongs to the service's lazy cache, not module import.
        with app.state.service_lock:
            if app.state.service is None:
                from .config import Settings
                from .service import RAGService

                app.state.service = RAGService(Settings.from_env())
        return app.state.service

    @app.middleware("http")
    async def request_log(request: Any, call_next: Any) -> Any:
        request_id = uuid4().hex
        started = perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        LOGGER.info("request_id=%s method=%s path=%s status=%d duration_ms=%.1f",
                    request_id, request.method, request.url.path,
                    response.status_code, (perf_counter() - started) * 1000)
        return response

    @app.exception_handler(ValueError)
    async def value_error(request: Any, exc: ValueError) -> JSONResponse:
        return JSONResponse({"detail": "参数或文档内容无效，请检查输入"}, status_code=400)

    @app.exception_handler(DocumentInputError)
    async def document_input_error(request: Any, exc: DocumentInputError) -> JSONResponse:
        return JSONResponse({"detail":exc.public_message,"code":exc.code},status_code=exc.status_code)

    @app.exception_handler(FileNotFoundError)
    async def file_not_found(request: Any, exc: FileNotFoundError) -> JSONResponse:
        return JSONResponse({"detail": "文档不存在"}, status_code=404)

    @app.exception_handler(RuntimeError)
    async def runtime_error(request: Any, exc: RuntimeError) -> JSONResponse:
        LOGGER.warning("service_unavailable exception_type=%s", type(exc).__name__)
        return JSONResponse({"detail": "模型或服务暂时不可用，请查看本地服务日志和配置"}, status_code=503)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Any, exc: Exception) -> JSONResponse:
        LOGGER.error("request_failed exception_type=%s", type(exc).__name__)
        return JSONResponse({"detail": "服务处理失败"}, status_code=500)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home() -> HTMLResponse:
        return HTMLResponse(Path(__file__).with_name("web.html").read_text(encoding="utf-8"))

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", **get_service().status()}

    @app.get("/documents")
    def list_documents() -> dict:
        return {"documents": get_service().list_documents()}

    @app.post("/documents", status_code=201)
    def upload_document(file: UploadFile = File(...)) -> dict:
        # Never join an uploaded name to a filesystem path at the API boundary.
        filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
        if not filename or Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
            raise HTTPException(400, "支持 PDF、Markdown、TXT、HTML 和 PNG/JPEG/WebP/BMP/TIFF 图片")
        content = file.file.read(max_upload_bytes + 1)
        if len(content) > max_upload_bytes:
            raise HTTPException(413, "文档超过大小限制")
        if not content:
            raise HTTPException(400, "不能上传空文档")
        return get_service().ingest_bytes(filename, content)

    @app.delete("/documents/{document_id}")
    def delete_document(document_id: str) -> dict:
        if not get_service().delete_document(document_id):
            raise HTTPException(404, "文档不存在")
        return {"deleted": True, "document_id": document_id}

    @app.get("/documents/{document_id}/file")
    def original_file(document_id: str) -> Response:
        original = get_service().get_original(document_id)
        if original is None:
            raise HTTPException(404, "原文件未保存或已删除；旧文档可重新上传以补存原文件")
        filename = original["filename"].replace("\\", "/").rsplit("/", 1)[-1]
        suffix = Path(filename).suffix.lower()
        is_pdf = suffix == ".pdf"
        media_type = "application/pdf" if is_pdf else "application/octet-stream"
        if suffix in IMAGE_SUFFIXES:
            # Decide MIME from the raster decoder rather than trusting an uploaded
            # extension. SVG/HTML are never rendered under this service's origin.
            from PIL import Image, UnidentifiedImageError
            try:
                with Image.open(BytesIO(original["content"])) as image:
                    media_type = IMAGE_MEDIA_TYPES.get(image.format, "")
                    image.verify()
                if not media_type:
                    raise ValueError("unsupported raster format")
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                raise HTTPException(400, "原图片格式无效或暂不支持预览") from exc
        disposition = "inline" if is_pdf or suffix in IMAGE_SUFFIXES else "attachment"
        fallback = "document.pdf" if is_pdf else "document" + (suffix if suffix in SUPPORTED_SUFFIXES else ".bin")
        return Response(
            original["content"],
            media_type=media_type,
            headers={
                "Content-Disposition": f"{disposition}; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename, safe='')}",
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "private, no-store",
                "Cross-Origin-Resource-Policy": "same-origin",
            },
        )

    @app.post("/query")
    def query(payload: QueryRequest) -> dict:
        return get_service().query(**payload.model_dump(exclude_none=True))

    @app.post("/agent")
    def agent(payload: AgentRequest) -> dict:
        return get_service().agent(**payload.model_dump(exclude_none=True))

    return app


app = create_app()
