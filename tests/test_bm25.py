from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracerag import BM25Retriever, Chunk, tokenize_bilingual, tokenize_char_ngrams  # noqa: E402


class BilingualTokenizerTests(unittest.TestCase):
    def test_normalizes_english_and_keeps_technical_terms(self) -> None:
        self.assertEqual(("bm25", "c++", "recall@k"), tokenize_bilingual("ＢＭ２５ C++ Recall@K"))

    def test_segments_chinese_into_words_without_fake_bigrams(self) -> None:
        self.assertEqual(("检索", "评估"), tokenize_bilingual("检索评估"))

    def test_tokenizes_mixed_chinese_and_english(self) -> None:
        self.assertEqual(
            ("稠密", "检索", "稠密检索", "dense"),
            tokenize_bilingual("稠密检索 Dense"),
        )

    def test_adds_rag_phrases_without_losing_component_words(self) -> None:
        tokens = tokenize_bilingual("文档切块与倒排索引")

        self.assertTrue({"文档", "切块", "倒排", "索引"}.issubset(tokens))
        self.assertTrue({"文档切块", "倒排索引"}.issubset(tokens))


class CharacterNGramTokenizerTests(unittest.TestCase):
    def test_generates_chinese_bigrams_and_trigrams(self) -> None:
        self.assertEqual(
            ("钙钛", "钛矿", "钙钛矿"),
            tokenize_char_ngrams("钙钛矿"),
        )

    def test_does_not_generate_ngrams_across_punctuation(self) -> None:
        tokens = tokenize_char_ngrams("检索，生成")

        self.assertEqual(("检索", "生成"), tokens)
        self.assertNotIn("索生", tokens)

    def test_keeps_english_technical_terms(self) -> None:
        self.assertEqual(
            ("crispr-cas9", "钙钛", "钛矿", "钙钛矿"),
            tokenize_char_ngrams("CRISPR-Cas9 钙钛矿"),
        )

    def test_rejects_invalid_ngram_sizes(self) -> None:
        with self.assertRaises(ValueError):
            tokenize_char_ngrams("检索", ())
        with self.assertRaises(ValueError):
            tokenize_char_ngrams("检索", (0, 2))


class BM25RetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = (
            Chunk("chunk-bm25", "doc-a", "BM25 uses term frequency and document frequency.", 0, 47, "a.md"),
            Chunk("chunk-dense", "doc-b", "Dense retrieval uses multilingual embeddings.", 0, 41, "b.md"),
            Chunk("chunk-citation", "doc-c", "Citations preserve source identifiers.", 0, 38, "c.md"),
        )
        self.retriever = BM25Retriever(self.chunks)

    def test_exact_terms_rank_relevant_chunk_first(self) -> None:
        results = self.retriever.retrieve("BM25 term frequency", top_k=2)

        self.assertEqual("chunk-bm25", results[0].chunk.chunk_id)
        self.assertEqual(1, results[0].rank)
        self.assertGreater(results[0].score, 0)
        self.assertEqual("a.md", results[0].chunk.source)

    def test_unseen_terms_return_no_results(self) -> None:
        self.assertEqual((), self.retriever.retrieve("quantum banana"))

    def test_top_k_larger_than_matches_returns_only_matches(self) -> None:
        results = self.retriever.retrieve("embeddings", top_k=10)
        self.assertEqual(("chunk-dense",), tuple(result.chunk.chunk_id for result in results))

    def test_chinese_query_matches_chinese_words(self) -> None:
        chunks = (
            Chunk("chunk-cn-retrieval", "doc-cn-a", "检索系统负责寻找相关证据。", 0, 13, "cn-a.md"),
            Chunk("chunk-cn-weather", "doc-cn-b", "杭州今天是晴朗天气。", 0, 11, "cn-b.md"),
        )
        results = BM25Retriever(chunks).retrieve("如何寻找检索证据？", top_k=1)
        self.assertEqual("chunk-cn-retrieval", results[0].chunk.chunk_id)

    def test_general_word_matches_document_domain_phrase(self) -> None:
        chunks = (
            Chunk("chunk-dense", "doc-a", "系统采用稠密检索。", 0, 9, "a.md"),
            Chunk("chunk-generation", "doc-b", "系统负责生成答案。", 0, 9, "b.md"),
        )

        results = BM25Retriever(chunks).retrieve("检索", top_k=1)

        self.assertEqual("chunk-dense", results[0].chunk.chunk_id)

    def test_character_ngrams_retrieve_unseen_domain_term(self) -> None:
        chunks = (
            Chunk("chunk-perovskite", "doc-a", "研究钙钛矿材料的晶体结构。", 0, 13, "a.md"),
            Chunk("chunk-graphene", "doc-b", "研究石墨烯材料的导电性能。", 0, 14, "b.md"),
        )
        retriever = BM25Retriever(chunks, tokenizer=tokenize_char_ngrams)

        results = retriever.retrieve("钙钛矿结构", top_k=1)

        self.assertEqual("chunk-perovskite", results[0].chunk.chunk_id)

    def test_equal_scores_are_ordered_by_chunk_id(self) -> None:
        chunks = (
            Chunk("chunk-b", "doc-b", "shared token", 0, 12),
            Chunk("chunk-a", "doc-a", "shared token", 0, 12),
        )
        results = BM25Retriever(chunks).retrieve("shared", top_k=2)
        self.assertEqual(("chunk-a", "chunk-b"), tuple(result.chunk.chunk_id for result in results))

    def test_invalid_query_and_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.retriever.retrieve("   ")
        with self.assertRaises(ValueError):
            self.retriever.retrieve("BM25", top_k=0)
        with self.assertRaises(ValueError):
            BM25Retriever(self.chunks, k1=0)
        with self.assertRaises(ValueError):
            BM25Retriever(self.chunks, b=1.1)
        with self.assertRaises(ValueError):
            BM25Retriever(self.chunks, k1=math.nan)


if __name__ == "__main__":
    unittest.main()
