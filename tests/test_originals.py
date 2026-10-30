from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from tracerag.schemas import Chunk, Document
from tracerag.storage import DocumentStore


def sample(source: str = "paper.pdf") -> tuple[Document,Chunk]:
    text = "Evidence from the original paper."
    return (
        Document("doc-a","Paper",text,source,2),
        Chunk("chunk-a","doc-a",text,0,len(text),source,2),
    )


class OriginalStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tracerag-original-test-")
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_original_bytes_survive_restarting_store_and_are_deleted_with_asset(self):
        store = DocumentStore(self.directory)
        document,chunk = sample()
        content = b"%PDF-1.7\noriginal-binary\x00\xff"
        record = store.put("asset-a","paper.pdf",len(content),(document,),(chunk,),original=content)
        self.assertTrue(record["original_available"])
        restored = DocumentStore(self.directory)
        self.assertEqual({"filename":"paper.pdf","content":content},restored.get_original("asset-a"))
        self.assertTrue(restored.delete("asset-a"))
        self.assertIsNone(restored.get_original("asset-a"))
        self.assertEqual((),restored.chunks())
        self.assertEqual({},restored.chunk_assets())
        self.assertEqual([],list(self.directory.glob("*.pdf")))

    def test_legacy_database_is_migrated_without_losing_records(self):
        with sqlite3.connect(self.directory/"tracerag.sqlite3") as db:
            db.execute("CREATE TABLE assets (id TEXT PRIMARY KEY, filename TEXT NOT NULL, bytes INTEGER NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
            db.execute("INSERT INTO assets(id,filename,bytes) VALUES ('old','old.pdf',42)")
        db.close()
        store = DocumentStore(self.directory)
        self.assertEqual("old.pdf",store.list_assets()[0]["filename"])
        self.assertFalse(store.list_assets()[0]["original_available"])
        self.assertIsNone(store.get_original("old"))
        self.assertIsNone(store.get_original("missing"))
        self.assertEqual(1,len(DocumentStore(self.directory).list_assets()))

    def test_duplicate_reupload_backfills_missing_original_only(self):
        store = DocumentStore(self.directory)
        document,chunk = sample()
        store.put("asset-a","paper.pdf",3,(document,),(chunk,))
        restored = store.put("asset-a","renamed.pdf",3,(document,),(chunk,),original=b"abc")
        self.assertTrue(restored["duplicate"])
        self.assertTrue(restored["original_available"])
        self.assertEqual({"filename":"paper.pdf","content":b"abc"},store.get_original("asset-a"))
        store.put("asset-a","third.pdf",3,(document,),(chunk,),original=b"bad")
        self.assertEqual(b"abc",store.get_original("asset-a")["content"])
        self.assertEqual(1,len(store.chunks()))

    def test_scope_filters_before_dedup_and_citation_mapping_uses_same_asset(self):
        store = DocumentStore(self.directory)
        document,first = sample("first.pdf")
        _,second = sample("second.pdf")
        store.put("asset-a","first.pdf",3,(document,),(first,),original=b"one")
        store.put("asset-b","second.pdf",3,(document,),(second,),original=b"two")
        self.assertEqual("first.pdf",store.chunks()[0].source)
        self.assertEqual({"chunk-a":"asset-a"},store.chunk_assets())
        self.assertEqual("second.pdf",store.chunks(["asset-b"])[0].source)
        self.assertEqual({"chunk-a":"asset-b"},store.chunk_assets(["asset-b"]))
        self.assertEqual((),store.chunks([]))
        self.assertEqual({},store.chunk_assets([]))
        self.assertEqual((),store.chunks(["missing"]))

    def test_scope_is_parameterized_and_never_interpreted_as_sql(self):
        store = DocumentStore(self.directory)
        document,chunk = sample()
        store.put("asset-a","paper.pdf",3,(document,),(chunk,))
        self.assertEqual((),store.chunks(["asset-a') OR 1=1 --"]))
        self.assertEqual(1,len(store.chunks()))
        with self.assertRaises(ValueError):
            store.chunks("asset-a")


if __name__ == "__main__":
    unittest.main()
