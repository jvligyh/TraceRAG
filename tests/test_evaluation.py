from __future__ import annotations

import unittest
import json
import hashlib
from pathlib import Path

from tracerag.evaluation import EvaluationCase, citation_id_precision, percentile, recall_at_k, reciprocal_rank, summarize


class RetrievalMetricsTests(unittest.TestCase):
    def test_recall_counts_unique_documents_and_multiple_relevant(self) -> None:
        self.assertEqual(0.5, recall_at_k(["a", "a", "b"], ["a", "c"], 2))
        self.assertEqual(1.0, recall_at_k(["a", "a", "c"], ["a", "c"], 2))

    def test_mrr_uses_first_relevant_document(self) -> None:
        self.assertEqual(0.5, reciprocal_rank(["x", "x", "b", "c"], ["b", "c"]))
        self.assertEqual(0.0, reciprocal_rank(["x"], ["b"]))

    def test_unanswerable_is_not_counted_as_perfect_recall(self) -> None:
        cases = [EvaluationCase("yes", "yes?", ("a",)), EvaluationCase("no", "no?", ())]
        report = summarize(cases, [("a",), ()], [0.1, 0.3])
        self.assertEqual(1.0, report["recall@1"])
        self.assertEqual(1, report["answerable_cases"])
        self.assertEqual(1.0, report["unanswerable_empty_retrieval_rate"])
        self.assertEqual(0, report["human_reviewed_cases"])

    def test_percentiles_interpolate(self) -> None:
        self.assertAlmostEqual(0.2, percentile([0.1, 0.3], 50))
        self.assertAlmostEqual(0.29, percentile([0.1, 0.3], 95))
        self.assertEqual(2, percentile([2], 95))

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            recall_at_k(["a"], [], 1)
        with self.assertRaises(ValueError):
            percentile([float("nan")], 50)
        with self.assertRaises(ValueError):
            summarize([], [], [])
        with self.assertRaises(ValueError):
            EvaluationCase("x", "q", ("a", "a"))

    def test_citation_ids_are_not_a_semantic_correctness_metric(self) -> None:
        self.assertEqual(0.5, citation_id_precision(["a", "a", "invented"], ["a"]))
        self.assertIsNone(citation_id_precision([], ["a"]))

    def test_seed_dataset_evidence_matches_stored_source_text(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cases = [json.loads(line) for line in (root / "data/evaluation/questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(40, len(cases))
        self.assertEqual(4, sum(not case["relevant_source_ids"] for case in cases))
        for case in cases:
            self.assertIsInstance(case["human_reviewed"], bool)
            if case["relevant_source_ids"]:
                source = case.get("evidence_source", case["relevant_source_ids"][0])
                self.assertIn(case["evidence"], (root / f"data/corpus/{source}.md").read_text(encoding="utf-8"))

    def test_public_corpus_manifest_matches_downloaded_bytes(self) -> None:
        corpus = Path(__file__).resolve().parents[1] / "data/corpus"
        manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(25, len(manifest["documents"]))
        self.assertEqual([], manifest["failures"])
        for record in manifest["documents"] + manifest.get("derived_documents", []):
            digest = hashlib.sha256((corpus / record["file"]).read_bytes()).hexdigest()
            self.assertEqual(record["sha256"], digest)
            self.assertEqual(record["document_id"], "md-" + digest[:16])


if __name__ == "__main__":
    unittest.main()
