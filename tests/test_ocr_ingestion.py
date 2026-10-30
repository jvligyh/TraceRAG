import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch

from PIL import Image

from tracerag.chunking import TextChunker
from tracerag.ingestion import PDFParser,parse_path
from tracerag.ocr import OCRResult
from tracerag.config import Settings
from tracerag.service import RAGService


class OCRIngestionTests(unittest.TestCase):
    def test_disguised_unsupported_image_is_rejected_before_ocr(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"renamed.png"
            with Image.new("RGB",(80,60),"white") as image:
                image.save(path,format="GIF")
            engine=Mock()
            with self.assertRaisesRegex(ValueError,"实际格式不支持"):
                parse_path(path,ocr_engine=engine)
            engine.recognize.assert_not_called()

    def test_image_text_and_confidence_propagate_to_chunks_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"image.png"
            with Image.new("RGB",(80,60),"white") as image:
                image.save(path)
            engine=Mock()
            engine.recognize.return_value=OCRResult("实验温度25 C\nEfficiency 23.5%",.94,2)
            document=parse_path(path,ocr_engine=engine)[0]
            self.assertEqual("ocr",document.extraction_method)
            chunk=TextChunker().chunk(document)[0]
            self.assertEqual(.94,chunk.ocr_confidence)
            settings=Settings(data_dir=Path(tmp)/"store",retrieval="bm25",translator="none")
            service=RAGService(settings,ocr_engine=engine)
            upload=service.ingest_bytes("image.png",path.read_bytes())
            self.assertTrue(upload["ocr_used"])
            self.assertEqual(1,upload["ocr_pages"])
            response=RAGService(settings).query("Efficiency",document_ids=[upload["document_id"]])
            self.assertEqual("ocr",response["citations"][0]["extraction_method"])
            self.assertEqual(.94,response["citations"][0]["ocr_confidence"])

    def test_mixed_pdf_only_ocr_sparse_pages_preserves_page_numbers_and_warns_blank(self):
        native="A real native PDF page has enough words to skip OCR."
        pages=[SimpleNamespace(extract_text=lambda:native),SimpleNamespace(extract_text=lambda:""),
               SimpleNamespace(extract_text=lambda:"")]
        engine=Mock()
        engine.recognize.side_effect=[OCRResult("Scanned page evidence",.9,1),OCRResult("",None,0)]
        warnings=[]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"mixed.pdf"; path.write_bytes(b"fixture")
            with patch("tracerag.ingestion.PdfReader",return_value=SimpleNamespace(pages=pages)), \
                 patch("tracerag.ocr.render_pdf_page",side_effect=lambda *a,**k:Image.new("RGB",(10,10))) as render:
                documents=PDFParser(enable_ocr=True,ocr_engine=engine,warnings=warnings).parse(path)
        self.assertEqual([1,2],[d.page_number for d in documents])
        self.assertEqual(["native","ocr"],[d.extraction_method for d in documents])
        self.assertEqual(2,render.call_count)
        self.assertIn("第3页",warnings[0])
        self.assertTrue(documents[1].document_id.endswith("-p0002-ocr"))

    def test_ocr_page_limit_fails_before_any_partial_recognition(self):
        pages=[SimpleNamespace(extract_text=lambda:"")]*3
        engine=Mock()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"scan.pdf"; path.write_bytes(b"fixture")
            with patch("tracerag.ingestion.PdfReader",return_value=SimpleNamespace(pages=pages)), \
                 self.assertRaisesRegex(ValueError,"上限2页"):
                PDFParser(enable_ocr=True,ocr_engine=engine,max_ocr_pages=2).parse(path)
        engine.recognize.assert_not_called()

    def test_no_text_invalid_images_and_disabled_ocr_are_not_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"empty.png"
            with Image.new("RGB",(80,60),"white") as image:
                image.save(path)
            engine=Mock(); engine.recognize.return_value=OCRResult("",None,0)
            service=RAGService(Settings(data_dir=Path(tmp)/"db"),ocr_engine=engine)
            for content in (path.read_bytes(),b"fakepng"):
                with self.assertRaises(ValueError):
                    service.ingest_bytes("empty.png",content)
            self.assertEqual([],service.list_documents())
            disabled=RAGService(Settings(data_dir=Path(tmp)/"off",ocr_enabled=False))
            with self.assertRaisesRegex(ValueError,"关闭"):
                disabled.ingest_bytes("empty.png",path.read_bytes())


if __name__=="__main__":
    unittest.main()
