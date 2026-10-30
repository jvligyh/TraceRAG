from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from tracerag.config import Settings
from tracerag.service import RAGService, _parse_answer
from tracerag.storage import CachedEmbeddingModel, DocumentStore


class CountingEmbedding:
    calls = 0
    def embed_passages(self,texts):
        self.calls += len(texts)
        return tuple((1.0,0.5) for _ in texts)
    def embed_query(self,text):
        return (1.0,0.5)


class GroundedClient:
    def complete(self,prompt):
        data=json.loads(prompt.split("DATA_JSON:\n",1)[1])
        return json.dumps({"answer":"RRF融合多个排名。", "citation_ids":[data["sources"][0]["chunk_id"]],
                           "insufficient_evidence":False})


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.cfg=Settings(data_dir=Path(self.temp.name),retrieval="bm25",translator="none")
        self.service=RAGService(self.cfg)
    def tearDown(self):
        self.temp.cleanup()

    def test_upload_restart_dedup_grounded_answer_and_delete(self):
        data=b"# RRF\nReciprocal rank fusion combines rankings."
        uploaded=self.service.ingest_bytes("../../paper.md",data)
        self.assertEqual("paper.md",uploaded["filename"])
        self.assertTrue(self.service.ingest_bytes("paper.md",data)["duplicate"])
        restarted=RAGService(self.cfg,client=GroundedClient())
        response=restarted.query("RRF fusion",1)
        self.assertEqual("answered",response["status"])
        self.assertEqual(response["results"][0]["chunk_id"],response["citations"][0]["chunk_id"])
        self.assertEqual("paper.md",response["citations"][0]["source"])
        self.assertTrue(restarted.delete_document(uploaded["document_id"]))
        self.assertEqual([],restarted.query("RRF",generate=False)["results"])

    def test_no_match_and_no_documents_do_not_call_model(self):
        self.assertEqual("insufficient_evidence",self.service.query("anything")["status"])
        self.service.ingest_bytes("a.md",b"retrieval generation")
        self.assertEqual("insufficient_evidence",self.service.query("xyz123")["status"])

    def test_html_strips_script_and_preserves_content(self):
        self.service.ingest_bytes("a.html",b"<h1>retrieval</h1><script>badword</script><p>ranking evidence</p>")
        response=self.service.query("retrieval",generate=False)
        self.assertIn("ranking evidence",response["results"][0]["text"])
        self.assertNotIn("badword",response["results"][0]["text"])

    def test_fabricated_citations_and_malformed_json_fail_closed(self):
        for raw in ('{}','not json','{"answer":"a","citation_ids":["fake"],"insufficient_evidence":false}'):
            with self.assertRaises(RuntimeError):
                _parse_answer(raw,{"real"})

    def test_embedding_cache_survives_restart_and_only_embeds_missing(self):
        model=CountingEmbedding()
        store=DocumentStore(Path(self.temp.name))
        cached=CachedEmbeddingModel(model,store,"test-v1")
        cached.embed_passages(("a","a","b"))
        self.assertEqual(2,model.calls)
        cached=CachedEmbeddingModel(model,DocumentStore(Path(self.temp.name)),"test-v1")
        cached.embed_passages(("a","c"))
        self.assertEqual(3,model.calls)
        CachedEmbeddingModel(model,store,"test-v2").embed_passages(("a",))
        self.assertEqual(4,model.calls)

    def test_invalid_upload_does_not_persist(self):
        with self.assertRaises(ValueError):
            self.service.ingest_bytes("a.exe",b"test")
        with self.assertRaises(ValueError):
            self.service.ingest_bytes("empty.md",b"")
        self.assertEqual([],self.service.list_documents())

    def test_identical_bytes_with_different_parsers_are_not_duplicates(self):
        content = b"<h1>retrieval</h1><p>evidence</p>"
        html = self.service.ingest_bytes("paper.html", content)
        markdown = self.service.ingest_bytes("paper.md", content)
        self.assertFalse(html["duplicate"])
        self.assertFalse(markdown["duplicate"])
        self.assertNotEqual(html["document_id"], markdown["document_id"])
        self.assertTrue(self.service.ingest_bytes("renamed.htm", content)["duplicate"])
        self.assertTrue(self.service.ingest_bytes("renamed.txt", content)["duplicate"])
        self.assertEqual(2, len(self.service.list_documents()))
        self.assertTrue(self.service.delete_document(html["document_id"]))
        self.assertEqual("paper.md", self.service.query("retrieval", generate=False)["results"][0]["source"])

    def test_truncated_citation_offsets_match_returned_text(self):
        cfg = Settings(data_dir=Path(self.temp.name), retrieval="bm25", translator="none", max_context_chars=15)
        service = RAGService(cfg)
        service.ingest_bytes("a.md", b"retrieval evidence and source offsets should remain consistent")
        citation = service.query("retrieval")["citations"][0]
        self.assertTrue(citation["truncated"])
        self.assertEqual(15, len(citation["text"]))
        self.assertEqual(len(citation["text"]), citation["end_char"] - citation["start_char"])

    def test_concurrent_store_put_is_idempotent_across_service_instances(self):
        services = [RAGService(self.cfg) for _ in range(4)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda s: s.ingest_bytes("a.md", b"retrieval evidence"), services))
        self.assertEqual(1, sum(not row["duplicate"] for row in results))
        self.assertEqual(1, len(self.service.list_documents()))

    def test_settings_context_environment_and_nonfinite_timeout(self):
        with patch.dict("os.environ", {"TRACERAG_MAX_CONTEXT_CHARS": "1234"}):
            self.assertEqual(1234, Settings.from_env().max_context_chars)
        for timeout in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                Settings(timeout_seconds=timeout)

    def test_invalid_embedding_does_not_poison_persistent_cache(self):
        store = DocumentStore(Path(self.temp.name))
        model = CountingEmbedding()
        cached = CachedEmbeddingModel(model, store, "test-invalid-v1")
        with patch.object(model, "embed_passages", return_value=((0.0, 0.0),)):
            with self.assertRaises(ValueError):
                cached.embed_passages(("evidence",))
        self.assertEqual(((1.0, 0.5),), cached.embed_passages(("evidence",)))

    def test_malformed_pdf_is_a_validation_error_and_does_not_persist(self):
        with self.assertRaises(ValueError):
            self.service.ingest_bytes("bad.pdf", b"this is not a PDF")
        self.assertEqual([], self.service.list_documents())


if __name__=="__main__":
    unittest.main()
