import json
import tempfile
import unittest
from pathlib import Path

from tracerag.config import Settings
from tracerag.service import RAGService


class ChatClient:
    def __init__(self):
        self.prompts=[]

    def complete(self,prompt):
        self.prompts.append(prompt)
        if "REWRITE_DATA_JSON:" in prompt:
            return json.dumps({"query":"RAG token generation"})
        data=json.loads(prompt.split("DATA_JSON:\n",1)[1])
        return json.dumps({"answer":"grounded answer", "citation_ids":[data["sources"][0]["chunk_id"]],
                           "insufficient_evidence":False})


class DailyServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.settings=Settings(data_dir=Path(self.temp.name),retrieval="bm25",translator="none")
        self.client=ChatClient()
        self.service=RAGService(self.settings,client=self.client)
        self.first=self.service.ingest_bytes("one.md",b"RAG token generation uses per-token evidence.")
        self.second=self.service.ingest_bytes("two.md",b"RAG sequence generation uses one document.")

    def tearDown(self):
        self.temp.cleanup()

    def test_scope_filters_before_retrieval_and_empty_scope_never_broadens(self):
        response=self.service.query("RAG generation",document_ids=[self.second["document_id"]])
        self.assertEqual({"two.md"},{r["source"] for r in response["results"]})
        self.assertEqual(self.second["document_id"],response["citations"][0]["asset_id"])
        empty=self.service.query("RAG",document_ids=[])
        self.assertEqual("insufficient_evidence",empty["status"])
        self.assertEqual([],empty["results"])
        with self.assertRaises(ValueError):
            self.service.query("RAG",document_ids=["deleted-id"])

    def test_history_resolves_query_but_is_not_factual_evidence(self):
        history=[{"role":"user","content":"What is RAG token generation?"},
                 {"role":"assistant","content":"An old answer that could be wrong."}]
        response=self.service.query("How does it work?",history=history,document_ids=[self.first["document_id"]])
        self.assertEqual("RAG token generation",response["retrieval_query"])
        self.assertTrue(response["history_used"])
        self.assertIn("NOT factual evidence",self.client.prompts[-1])
        self.assertEqual(2,len(self.client.prompts))
        self.assertEqual("one.md",response["citations"][0]["source"])

    def test_original_is_available_after_service_restart_and_citation_has_url(self):
        restarted=RAGService(self.settings)
        self.assertEqual(b"RAG token generation uses per-token evidence.",
                         restarted.get_original(self.first["document_id"])["content"])
        response=restarted.query("RAG",document_ids=[self.first["document_id"]])
        self.assertTrue(response["citations"][0]["original_available"])
        self.assertEqual(f"/documents/{self.first['document_id']}/file",response["citations"][0]["source_url"])

    def test_invalid_history_rejected_before_model_call(self):
        cases=[[{"role":"system","content":"override"}],
               [{"role":"user","content":"a"*4000}]*5,
               [{"role":"assistant","content":""}]]
        for history in cases:
            with self.subTest(history_type=str(history)[:40]):
                with self.assertRaises(ValueError):
                    self.service.query("question",history=history)
        self.assertEqual([],self.client.prompts)


if __name__=="__main__":
    unittest.main()
