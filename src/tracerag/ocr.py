"""本地中英文文字识别与扫描PDF渲染，不把OCR当成图片语义理解。"""
from __future__ import annotations

import hashlib
import importlib.metadata
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


_PDFIUM_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class OCRResult:
    text: str
    mean_confidence: float | None
    lines: int


class RapidOCREngine:
    """懒加载的RapidOCR/PP-OCRv6 small CPU适配器，同一实例串行推理。"""

    MODEL_FAMILY = "PP-OCRv6-small"

    def __init__(self, cache_dir: Path | None = None, min_confidence: float = 0.5,
                 max_image_pixels: int = 20_000_000) -> None:
        if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
            raise ValueError("OCR min_confidence must be between 0 and 1")
        if isinstance(max_image_pixels, bool) or not isinstance(max_image_pixels, int) or max_image_pixels < 1:
            raise ValueError("OCR max_image_pixels must be a positive integer")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path.home() / ".cache/tracerag/ocr"
        self.min_confidence = float(min_confidence)
        self.max_image_pixels = max_image_pixels
        self._engine: Any = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    def _create_engine(self) -> Any:
        try:
            from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR
        except ImportError as exc:
            raise RuntimeError('缺少本地OCR依赖，请在项目目录运行 python -m pip install -e .') from exc
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            return RapidOCR(params={
                "Global.model_root_dir": str(self.cache_dir),
                "Global.log_level": "warning",
                "Global.text_score": self.min_confidence,
                "Det.engine_type": EngineType.ONNXRUNTIME,
                "Det.ocr_version": OCRVersion.PPOCRV6,
                "Det.model_type": ModelType.SMALL,
                "Cls.engine_type": EngineType.ONNXRUNTIME,
                "Cls.ocr_version": OCRVersion.PPOCRV4,
                "Cls.model_type": ModelType.MOBILE,
                "Rec.engine_type": EngineType.ONNXRUNTIME,
                "Rec.ocr_version": OCRVersion.PPOCRV6,
                "Rec.model_type": ModelType.SMALL,
                "EngineConfig.onnxruntime.use_cuda": False,
                "EngineConfig.onnxruntime.use_dml": False,
                "EngineConfig.onnxruntime.intra_op_num_threads": 2,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            })
        except Exception as exc:
            raise RuntimeError("本地OCR初始化失败，请检查依赖和模型缓存") from exc

    def recognize(self, image: Image.Image) -> OCRResult:
        if not isinstance(image, Image.Image):
            raise ValueError("OCR input must be a Pillow image")
        width, height = image.size
        if min(width, height) < 1 or width * height > self.max_image_pixels:
            raise ValueError("图片像素超过OCR限制，请缩小图片后重试")
        prepared = None
        oriented = None
        try:
            oriented = ImageOps.exif_transpose(image)
            # PIL输入被RapidOCR解释为RGB；ndarray输入会按BGR解释，故不传ndarray。
            if oriented.mode in {"RGBA", "LA"} or (oriented.mode == "P" and "transparency" in oriented.info):
                rgba = oriented.convert("RGBA")
                try:
                    prepared = Image.new("RGB", rgba.size, "white")
                    prepared.paste(rgba, mask=rgba.getchannel("A"))
                finally:
                    rgba.close()
            else:
                prepared = oriented.convert("RGB")
            prepared.load()
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            if prepared is not None:
                prepared.close()
            if oriented is not None and oriented is not image:
                oriented.close()
            raise ValueError("无法读取此图片，请确认文件有效") from exc
        try:
            with self._lock:
                if self._engine is None:
                    self._engine = self._create_engine()
                try:
                    result = self._engine(prepared)
                except Exception as exc:
                    raise RuntimeError("本地OCR识别失败，请检查图片和模型缓存") from exc
            return _parse_output(result, self.min_confidence)
        finally:
            if prepared is not None:
                prepared.close()
            if oriented is not None and oriented is not image:
                oriented.close()

    def metadata(self) -> dict:
        """不加载模型；记录已下载的实际模型哈希，便于复现。"""
        try:
            version = importlib.metadata.version("rapidocr")
        except importlib.metadata.PackageNotFoundError:
            version = None
        files = []
        if self.cache_dir.is_dir():
            for path in sorted(self.cache_dir.rglob("*.onnx")):
                with path.open("rb") as source:
                    digest = hashlib.file_digest(source, "sha256").hexdigest()
                files.append({"file": path.name, "bytes": path.stat().st_size, "sha256": digest})
        return {"engine": "RapidOCR", "version": version, "model_family": self.MODEL_FAMILY,
                "device": "cpu", "min_confidence": self.min_confidence, "models": files}


def _parse_output(result: Any, min_confidence: float) -> OCRResult:
    texts, scores = getattr(result, "txts", None), getattr(result, "scores", None)
    if texts is None and scores is None:
        return OCRResult("", None, 0)
    try:
        if isinstance(texts, (str, bytes)) or isinstance(scores, (str, bytes)) or texts is None or scores is None:
            raise ValueError("invalid OCR arrays")
        if len(texts) != len(scores):
            raise ValueError("OCR text/score count mismatch")
        lines, confidences = [], []
        for text, score in zip(texts, scores, strict=True):
            confidence = float(score)
            if not isinstance(text, str) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("invalid OCR line")
            if text.strip() and confidence >= min_confidence:
                lines.append(text.strip())
                confidences.append(confidence)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("OCR模型返回了无效的文字或置信度") from exc
    return OCRResult("\n".join(lines), math.fsum(confidences) / len(confidences) if confidences else None,
                     len(lines))


def render_pdf_page(path: str | Path, page_number: int, scale: float = 2,
                    max_pixels: int = 20_000_000) -> Image.Image:
    """将单页PDF转为独立RGB图片。PDFium全局锁避免跨实例线程同时进入原生库。"""
    if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:
        raise ValueError("PDF page_number must be a positive integer")
    if not math.isfinite(scale) or not 0 < scale <= 4 or max_pixels < 1:
        raise ValueError("invalid PDF OCR render limits")
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError('缺少扫描PDF渲染依赖，请在项目目录运行 python -m pip install -e .') from exc
    pdf = page = bitmap = view = None
    with _PDFIUM_LOCK:
        try:
            pdf = pdfium.PdfDocument(str(path))
            if page_number > len(pdf):
                raise ValueError("PDF page_number exceeds page count")
            page = pdf[page_number - 1]
            width, height = page.get_size()
            if (not math.isfinite(width) or not math.isfinite(height) or min(width, height) <= 0
                    or math.ceil(width * scale) * math.ceil(height * scale) > max_pixels):
                raise ValueError("PDF页面渲染像素超过OCR限制")
            bitmap = page.render(scale=scale)
            view = bitmap.to_pil()
            return view.convert("RGB")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("无法渲染此PDF页面，请检查文件是否有效或加密") from exc
        finally:
            # Image must be detached before releasing native bitmap memory.
            if view is not None:
                view.close()
            for resource in (bitmap, page, pdf):
                if resource is not None:
                    resource.close()
