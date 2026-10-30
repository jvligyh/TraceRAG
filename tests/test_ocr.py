from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from tracerag.ocr import OCRResult, RapidOCREngine, _parse_output, render_pdf_page


class OCREngineTests(unittest.TestCase):
    def test_filters_low_confidence_blank_and_returns_mean(self):
        result = _parse_output(SimpleNamespace(txts=(" 中文 ", "noise", "", "English"),
                                              scores=(0.9, 0.2, 1.0, 0.7)), 0.5)
        self.assertEqual("中文\nEnglish", result.text)
        self.assertEqual(2, result.lines)
        self.assertAlmostEqual(0.8, result.mean_confidence)

    def test_empty_or_all_low_confidence_has_no_text(self):
        self.assertEqual(OCRResult("", None, 0), _parse_output(SimpleNamespace(txts=None, scores=None), .5))
        self.assertEqual(OCRResult("", None, 0), _parse_output(SimpleNamespace(txts=["bad"], scores=[.1]), .5))

    def test_invalid_output_fails_instead_of_becoming_blank_page(self):
        for texts, scores in (("text", [1]), (["x"], None), (["x"], []), (["x"], [float("nan")]),
                              (["x"], [2]), ([42], [.9])):
            with self.subTest(texts=texts, scores=scores), self.assertRaises(RuntimeError):
                _parse_output(SimpleNamespace(txts=texts, scores=scores), .5)

    def test_engine_loads_only_once_and_passes_rgb_pillow(self):
        engine = RapidOCREngine()
        self.assertFalse(engine.loaded)
        received = []
        def fake(image):
            received.append((image.mode, image.getpixel((0, 0))))
            return SimpleNamespace(txts=["a"], scores=[.9])
        with Image.new("RGBA", (30, 30), (255, 0, 0, 0)) as image:
            with patch.object(engine, "_create_engine", return_value=fake) as factory:
                engine.recognize(image)
                engine.recognize(image)
                factory.assert_called_once()
                self.assertTrue(engine.loaded)
            self.assertEqual("RGBA", image.mode)
        self.assertEqual([("RGB", (255, 255, 255)), ("RGB", (255, 255, 255))], received)

    def test_large_or_wrong_input_rejected_before_loading(self):
        engine = RapidOCREngine(max_image_pixels=100)
        with patch.object(engine, "_create_engine") as factory:
            with Image.new("RGB", (11, 10)) as image, self.assertRaises(ValueError):
                engine.recognize(image)
            with self.assertRaises(ValueError):
                engine.recognize(b"not an image")
            factory.assert_not_called()

    def test_invalid_limits_are_rejected(self):
        for confidence in (-1, 2, float("nan")):
            with self.assertRaises(ValueError):
                RapidOCREngine(min_confidence=confidence)
        with self.assertRaises(ValueError):
            RapidOCREngine(max_image_pixels=0)

    def test_model_errors_are_sanitized(self):
        engine = RapidOCREngine()
        engine._engine = Mock(side_effect=RuntimeError("private exception details"))
        with Image.new("RGB", (20, 20)) as image, self.assertRaises(RuntimeError) as caught:
            engine.recognize(image)
        self.assertNotIn("private", str(caught.exception))

    def test_exif_orientation_is_applied_without_mutating_original(self):
        engine = RapidOCREngine()
        sizes = []
        engine._engine = lambda image: (sizes.append(image.size) or SimpleNamespace(txts=None, scores=None))
        with Image.new("RGB", (30, 50)) as image:
            image.getexif()[274] = 6
            engine.recognize(image)
            self.assertEqual((30, 50), image.size)
        self.assertEqual([(50, 30)], sizes)

    def test_missing_dependency_has_actionable_install_message(self):
        engine = RapidOCREngine()
        with patch.dict(sys.modules, {"rapidocr": None}), Image.new("RGB", (20, 20)) as image:
            with self.assertRaisesRegex(RuntimeError, "pip install -e"):
                engine.recognize(image)


class PDFRenderTests(unittest.TestCase):
    def fake_pdf(self, size=(100, 100)):
        image = Image.new("RGB", (200, 200), "white")
        bitmap = Mock()
        bitmap.to_pil.return_value = image
        page = Mock()
        page.get_size.return_value = size
        page.render.return_value = bitmap
        class FakePDF:
            def __len__(self): return 1
            def __getitem__(self, index): return page
            close = Mock()
        document = FakePDF()
        module = SimpleNamespace(PdfDocument=Mock(return_value=document))
        return module, document, page, bitmap

    def test_render_copies_image_before_closing_native_resources(self):
        module, pdf, page, bitmap = self.fake_pdf()
        with patch.dict(sys.modules, {"pypdfium2": module}):
            rendered = render_pdf_page("paper.pdf", 1)
        self.assertEqual("RGB", rendered.mode)
        self.assertEqual((255, 255, 255), rendered.getpixel((0, 0)))
        page.render.assert_called_once_with(scale=2)
        bitmap.close.assert_called_once()
        page.close.assert_called_once()
        pdf.close.assert_called_once()
        rendered.close()

    def test_render_limit_rejects_before_allocating_bitmap(self):
        module, pdf, page, _ = self.fake_pdf(size=(5000, 5000))
        with patch.dict(sys.modules, {"pypdfium2": module}), self.assertRaises(ValueError):
            render_pdf_page("paper.pdf", 1, max_pixels=1000)
        page.render.assert_not_called()
        page.close.assert_called_once()
        pdf.close.assert_called_once()

    def test_out_of_bounds_page_closes_pdf(self):
        module, pdf, _, _ = self.fake_pdf()
        with patch.dict(sys.modules, {"pypdfium2": module}), self.assertRaises(ValueError):
            render_pdf_page("paper.pdf", 2)
        pdf.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
