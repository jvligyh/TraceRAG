import unittest
import json
import tempfile
from pathlib import Path

from tracerag.reranking import CrossEncoderReranker, NoOpReranker
from tracerag.schemas import Chunk, RetrievedChunk
from tracerag.transformer_embeddings import TransformerEmbeddingModel, _complete_safetensors


def result(cid, rank):
    return RetrievedChunk(Chunk(cid, "doc", cid, 0, len(cid)), .5, rank)


class ScoredReranker(CrossEncoderReranker):
    def _score(self, query, texts):
        return [1.0 if text == "a" else 3.0 for text in texts]


class RerankerTests(unittest.TestCase):
    def test_reranker_changes_order_and_assigns_new_ranks(self):
        rows = ScoredReranker().rerank("q", [result("a", 1), result("b", 2)], top_k=1)
        self.assertEqual("b", rows[0].chunk.chunk_id)
        self.assertEqual(1, rows[0].rank)
        self.assertEqual(3.0, rows[0].score)

    def test_empty_and_duplicate_candidates(self):
        model = ScoredReranker()
        self.assertEqual((), model.rerank("q", []))
        with self.assertRaises(ValueError):
            model.rerank("q", [result("a", 1), result("a", 2)])

    def test_noop_preserves_candidates(self):
        rows = (result("a", 1), result("b", 2))
        self.assertEqual(rows[:1], NoOpReranker().rerank("q", rows, 1))

    def test_invalid_model_scores_do_not_escape(self):
        model = ScoredReranker()
        model._score = lambda query, texts: [float("nan")]
        with self.assertRaises(RuntimeError):
            model.rerank("q", [result("a", 1)])
        model._score = lambda query, texts: []
        with self.assertRaises(RuntimeError):
            model.rerank("q", [result("a", 1)])

    def test_embedding_load_is_lazy_and_empty_query_rejected(self):
        model = TransformerEmbeddingModel()
        self.assertIsNone(model._model)
        self.assertEqual((), model.embed_passages([]))
        with self.assertRaises(ValueError):
            model.embed_query(" ")

    def test_partial_weight_file_is_not_a_complete_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            header = json.dumps({"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
            path.write_bytes(len(header).to_bytes(8, "little") + header + bytes(4))
            self.assertFalse(_complete_safetensors(path))
            path.write_bytes(len(header).to_bytes(8, "little") + header + bytes(8))
            self.assertTrue(_complete_safetensors(path))


if __name__ == "__main__":
    unittest.main()
