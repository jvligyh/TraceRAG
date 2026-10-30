from __future__ import annotations

import hashlib
import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol

from pypdf import PdfReader

from .schemas import Document
from .errors import DocumentInputError


class DocumentParser(Protocol):
    """文档解析器的统一接口。"""

    def parse(self, path: Path) -> tuple[Document, ...]:
        """解析一个文件，返回一个或多个文档。"""


class MarkdownParser:
    """将一个 Markdown 文件解析为一个文档。"""

    def parse(self, path: Path) -> tuple[Document, ...]:
        path = _require_file(path)
        raw = path.read_bytes()
        text = normalize_text(raw.decode("utf-8-sig"))
        title = _markdown_title(text) or path.stem
        document_id = f"md-{_short_hash(raw)}"
        return (
            Document(
                document_id=document_id,
                title=title,
                text=text,
                source=str(path.resolve()),
            ),
        )


class PDFParser:
    """优先提取文字层；启用OCR时对缺少文字的页面逐页识别。"""

    def __init__(self, *, enable_ocr: bool = False, ocr_engine=None, max_ocr_pages: int = 30,
                 max_image_pixels: int = 20_000_000, warnings: list[str] | None = None) -> None:
        self.enable_ocr,self.ocr_engine=enable_ocr,ocr_engine
        self.max_ocr_pages,self.max_image_pixels=max_ocr_pages,max_image_pixels
        self.warnings=warnings if warnings is not None else []

    def parse(self, path: Path) -> tuple[Document, ...]:
        path = _require_file(path)
        file_hash = _short_hash(path.read_bytes())
        reader = PdfReader(path)
        documents: list[Document] = []

        native_texts=[normalize_text(page.extract_text() or "") for page in reader.pages]
        needs_ocr=[i for i,text in enumerate(native_texts) if len(re.sub(r"\s","",text))<20]
        if self.enable_ocr and len(needs_ocr)>self.max_ocr_pages:
            raise DocumentInputError(f"PDF有{len(needs_ocr)}页需要OCR，当前上限{self.max_ocr_pages}页；请拆分PDF后上传。", "ocr_page_limit")
        if self.enable_ocr and needs_ocr:
            from .ocr import RapidOCREngine,render_pdf_page
            engine=self.ocr_engine or RapidOCREngine(max_image_pixels=self.max_image_pixels)
        for page_number, native_text in enumerate(native_texts,start=1):
            text=native_text
            method,confidence="native",None
            if self.enable_ocr and page_number-1 in needs_ocr:
                with render_pdf_page(path,page_number,max_pixels=self.max_image_pixels) as image:
                    recognized=engine.recognize(image)
                recognized_text=normalize_text(recognized.text)
                if recognized_text:
                    text,method,confidence=recognized_text,"ocr",recognized.mean_confidence
                else:
                    if re.search(r"[A-Za-z\u4e00-\u9fff]{2}",native_text):
                        self.warnings.append(f"第{page_number}页OCR未识别到更多文字，已保留原生文字。")
                    else:
                        self.warnings.append(f"第{page_number}页OCR未识别到可用文字；该页未建立检索索引。")
                        # 单独页码不应冒充识别成功的文献正文。
                        continue
            if not text:
                continue
            documents.append(
                Document(
                    document_id=f"pdf-{file_hash}-p{page_number:04d}" + ("-ocr" if method=="ocr" else ""),
                    title=f"{path.stem}（第 {page_number} 页）",
                    text=text,
                    source=str(path.resolve()),
                    page_number=page_number,
                    extraction_method=method,
                    ocr_confidence=confidence,
                )
            )

        if not documents:
            raise DocumentInputError("PDF未提取或识别到可用文字；请检查扫描清晰度或启用OCR。", "ocr_empty")
        return tuple(documents)


class ImageParser:
    """识别单张光栅图片中的文字；图片仍作为原文件供核对。"""

    def __init__(self, ocr_engine=None, max_image_pixels: int = 20_000_000) -> None:
        self.ocr_engine,self.max_image_pixels=ocr_engine,max_image_pixels

    def parse(self, path: Path) -> tuple[Document,...]:
        from PIL import Image,UnidentifiedImageError
        from .ocr import RapidOCREngine
        path=_require_file(path)
        try:
            with Image.open(path) as image:
                if image.format not in {"PNG", "JPEG", "WEBP", "BMP", "TIFF"}:
                    raise DocumentInputError("图片实际格式不支持，请转换成PNG/JPEG/WebP/BMP/TIFF。", "invalid_image")
                if image.width*image.height>self.max_image_pixels:
                    raise DocumentInputError("图片像素过大，请缩小后重试。", "image_pixel_limit",413)
                if getattr(image,"n_frames",1)>1:
                    raise DocumentInputError("多帧图片请转换成PDF或拆成单张图片后上传。", "multi_frame_image")
                result=(self.ocr_engine or RapidOCREngine(max_image_pixels=self.max_image_pixels)).recognize(image)
        except (UnidentifiedImageError,OSError,Image.DecompressionBombError) as exc:
            raise DocumentInputError("无法读取图片，请上传有效的PNG/JPEG/WebP/BMP/TIFF。", "invalid_image") from exc
        text=normalize_text(result.text)
        if not text:
            raise DocumentInputError("图片中未识别到可用文字；请上传清晰截图或扫描图。", "ocr_empty")
        return (Document(document_id="img-"+_short_hash(path.read_bytes())+"-ocr",title=path.stem,text=text,
                         source=str(path.resolve()),extraction_method="ocr",ocr_confidence=result.mean_confidence),)


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in {"script", "style", "noscript"}:
            self.hidden += 1
        elif tag in {"p", "div", "br", "h1", "h2", "h3", "li", "pre", "tr"} and not self.hidden:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"}:
            self.hidden = max(0,self.hidden-1)
        elif tag in {"p", "div", "h1", "h2", "h3", "li", "pre", "tr"} and not self.hidden:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


class HTMLParserDocument:
    """仅提取HTML可见文本，不执行脚本、不拉取远端资源。"""

    def parse(self, path: Path) -> tuple[Document,...]:
        path = _require_file(path)
        raw = path.read_bytes()
        parser = _HTMLText()
        parser.feed(raw.decode("utf-8-sig"))
        text = normalize_text("".join(parser.parts))
        return (Document("html-"+_short_hash(raw),path.stem,text,str(path.resolve())),)


def parse_path(path: str | Path, *, enable_ocr: bool = True, ocr_engine=None,
               max_ocr_pages: int = 30, max_image_pixels: int = 20_000_000,
               warnings: list[str] | None = None) -> tuple[Document, ...]:
    """根据文件后缀选择解析器。"""

    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        return MarkdownParser().parse(resolved)
    if suffix in {".html", ".htm"}:
        return HTMLParserDocument().parse(resolved)
    if suffix == ".pdf":
        return PDFParser(enable_ocr=enable_ocr,ocr_engine=ocr_engine,max_ocr_pages=max_ocr_pages,
                         max_image_pixels=max_image_pixels,warnings=warnings).parse(resolved)
    if suffix in {".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff"}:
        if not enable_ocr:
            raise DocumentInputError("图片识别已关闭；请启用TRACERAG_OCR_ENABLED。", "ocr_disabled")
        return ImageParser(ocr_engine,max_image_pixels).parse(resolved)
    raise ValueError(f"unsupported document type: {suffix or '<none>'}")


def normalize_text(text: str) -> str:
    """统一 Unicode、换行符和过多空行，保留代码缩进。"""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"document file not found: {path}")
    return path


def _markdown_title(text: str) -> str | None:
    for line in text.splitlines():
        match = re.match(r"^\s*#\s+(.+?)\s*$", line)
        if match:
            return match.group(1)
    return None


def _short_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]
