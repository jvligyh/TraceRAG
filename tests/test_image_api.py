from __future__ import annotations

import unittest
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient
from PIL import Image

from tracerag.api import create_app
from tracerag.config import Settings
from tracerag.ocr import OCRResult
from tracerag.service import RAGService


def image_bytes(format: str = "PNG") -> bytes:
    buffer = BytesIO()
    with Image.new("RGB", (32, 12), "white") as image:
        image.save(buffer, format=format)
    return buffer.getvalue()


class ImageService:
    """Only an HTTP-boundary fake; it never loads an OCR model."""

    def __init__(self) -> None:
        self.uploads: list[tuple[str, bytes]] = []

    def status(self) -> dict:
        return {"provider": "extractive", "documents": 0, "ocr_enabled": True, "ocr_loaded": False}

    def list_documents(self) -> list:
        return []

    def ingest_bytes(self, filename: str, content: bytes) -> dict:
        self.uploads.append((filename, content))
        if content == b"invalid-image":
            raise ValueError("private decoder details")
        return {"document_id": "scan-a", "ocr_used": True, "ocr_pages": 2,
                "warnings": ["第2页识别置信度较低，请核对原图。"], "extraction_methods": ["ocr"]}

    def get_original(self, document_id: str) -> dict | None:
        samples = {
            "png": ("中文截图.png", image_bytes()),
            "mismatched-extension": ("photo.jpg", image_bytes()),
            "jpeg": ("photo.jpeg", image_bytes("JPEG")),
            "webp": ("photo.webp", image_bytes("WEBP")),
            "bmp": ("photo.bmp", image_bytes("BMP")),
            "tiff": ("scan.tiff", image_bytes("TIFF")),
            "gif-disguised": ("scan.png", image_bytes("GIF")),
            "corrupted": ("scan.png", b"<script>not an image</script>"),
            "html": ("page.html", b"<script>not inline</script>"),
        }
        if document_id not in samples:
            return None
        filename, content = samples[document_id]
        return {"filename": filename, "content": content}


class ImageApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ImageService()
        self.client = TestClient(create_app(self.service))

    def test_raster_upload_extensions_are_accepted_and_ocr_metadata_preserved(self) -> None:
        for extension in ("png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"):
            with self.subTest(extension=extension):
                response = self.client.post("/documents", files={"file": (f"scan.{extension}", image_bytes())})
                self.assertEqual(201, response.status_code, response.text)
                self.assertTrue(response.json()["ocr_used"])
                self.assertEqual(2, response.json()["ocr_pages"])
                self.assertIn("较低", response.json()["warnings"][0])
        self.assertEqual(7, len(self.service.uploads))

    def test_scanned_pdf_metadata_is_not_lost_by_http_boundary(self) -> None:
        response = self.client.post("/documents", files={"file": ("scan.pdf", b"%PDF-fake-for-boundary")})
        self.assertEqual(201, response.status_code)
        self.assertEqual(["ocr"], response.json()["extraction_methods"])
        self.assertEqual(1, len(response.json()["warnings"]))

    def test_svg_is_rejected_before_reaching_service(self) -> None:
        response = self.client.post("/documents", files={"file": ("image.svg", b"<svg><script/></svg>")})
        self.assertEqual(400, response.status_code)
        self.assertEqual([], self.service.uploads)

    def test_invalid_image_service_error_is_safe_validation_response(self) -> None:
        response = self.client.post("/documents", files={"file": ("image.png", b"invalid-image")})
        self.assertEqual(400, response.status_code)
        self.assertNotIn("decoder", response.text)

    def test_original_raster_bytes_inline_with_decoder_verified_mime(self) -> None:
        for document_id, expected in (("png", "image/png"), ("mismatched-extension", "image/png"),
                                      ("jpeg", "image/jpeg"), ("webp", "image/webp"),
                                      ("bmp", "image/bmp"), ("tiff", "image/tiff")):
            with self.subTest(document_id=document_id):
                response = self.client.get(f"/documents/{document_id}/file")
                self.assertEqual(200, response.status_code, response.text[:100])
                self.assertEqual(expected, response.headers["content-type"])
                self.assertTrue(response.headers["content-disposition"].startswith("inline;"))
                self.assertEqual("nosniff", response.headers["x-content-type-options"])
                self.assertEqual(self.service.get_original(document_id)["content"], response.content)

    def test_unapproved_or_corrupt_raster_is_not_served_inline(self) -> None:
        self.assertEqual(400, self.client.get("/documents/gif-disguised/file").status_code)
        self.assertEqual(400, self.client.get("/documents/corrupted/file").status_code)
        response = self.client.get("/documents/html/file")
        self.assertTrue(response.headers["content-disposition"].startswith("attachment;"))
        self.assertEqual("application/octet-stream", response.headers["content-type"])
        self.assertEqual(404, self.client.get("/documents/missing/file").status_code)

    def test_upload_ui_explains_ocr_limits_and_accepts_images(self) -> None:
        response = self.client.get("/")
        self.assertIn(".png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff", response.text)
        self.assertIn("OCR识别文本，请核对原图", response.text)
        self.assertIn("不等于正确率", response.text)

    def test_empty_ocr_explains_how_to_retry_without_storing_blank_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = Mock()
            engine.recognize.return_value = OCRResult("", None, 0)
            service = RAGService(Settings(data_dir=Path(tmp), retrieval="bm25", translator="none"),
                                 ocr_engine=engine)
            client = TestClient(create_app(service))
            response = client.post("/documents", files={"file": ("blank.png", image_bytes())})
            self.assertEqual(400, response.status_code)
            self.assertEqual("ocr_empty", response.json()["code"])
            self.assertIn("清晰截图", response.json()["detail"])
            self.assertEqual([], service.list_documents())

    def test_large_image_fails_before_ocr_with_actionable_pixel_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = Mock()
            service = RAGService(Settings(data_dir=Path(tmp), ocr_max_image_pixels=100_000), ocr_engine=engine)
            buffer = BytesIO()
            with Image.new("RGB", (400, 300), "white") as image:
                image.save(buffer, format="PNG")
            response = TestClient(create_app(service)).post(
                "/documents", files={"file": ("large.png", buffer.getvalue())})
            self.assertEqual(413, response.status_code)
            self.assertEqual("image_pixel_limit", response.json()["code"])
            self.assertIn("缩小", response.json()["detail"])
            engine.recognize.assert_not_called()
            self.assertEqual([], service.list_documents())


if __name__ == "__main__":
    unittest.main()
