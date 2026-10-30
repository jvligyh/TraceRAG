from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

from .schemas import Chunk, Document
from .retrieval import _validate_embedding_batch


class DocumentStore:
    """SQLite事务保存解析文档、切块与向量缓存；上传原文件不进入Git。"""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "tracerag.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS assets (
                    id TEXT PRIMARY KEY, filename TEXT NOT NULL, bytes INTEGER NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP, original BLOB);
                CREATE TABLE IF NOT EXISTS documents (
                    asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
                    id TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(asset_id,id));
                CREATE TABLE IF NOT EXISTS chunks (
                    asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
                    id TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(asset_id,id));
                CREATE TABLE IF NOT EXISTS vectors (
                    model TEXT NOT NULL, text_hash TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(model,text_hash));
            """)
            # 0.2.0 的旧数据库只存解析文本；无损添加原文件列，不伪造旧文件。
            if "original" not in {row[1] for row in db.execute("PRAGMA table_info(assets)")}:
                try:
                    db.execute("ALTER TABLE assets ADD COLUMN original BLOB")
                except sqlite3.OperationalError:
                    # 两个服务恰好同时打开旧库时，另一连接可能已经完成迁移。
                    if "original" not in {row[1] for row in db.execute("PRAGMA table_info(assets)")}:
                        raise

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def put(self, asset_id: str, filename: str, size: int,
            documents: Sequence[Document], chunks: Sequence[Chunk], *, original: bytes | None = None) -> dict:
        with self.connect() as db:
            inserted = db.execute("INSERT OR IGNORE INTO assets(id,filename,bytes,original) VALUES (?,?,?,?)",
                                  (asset_id,filename,size,original)).rowcount > 0
            if inserted:
                db.executemany("INSERT INTO documents VALUES (?,?,?)", [
                    (asset_id,d.document_id,json.dumps(asdict(d),ensure_ascii=False)) for d in documents])
                db.executemany("INSERT INTO chunks VALUES (?,?,?)", [
                    (asset_id,c.chunk_id,json.dumps(asdict(c),ensure_ascii=False)) for c in chunks])
            elif original is not None:
                # 旧记录重新上传相同内容时只补原文件，不覆盖旧来源名或切块快照。
                db.execute("UPDATE assets SET original=? WHERE id=? AND original IS NULL", (original,asset_id))
        item = next(x for x in self.list_assets() if x["document_id"] == asset_id)
        return {**item, "duplicate": not inserted}

    def list_assets(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("""SELECT a.id,a.filename,a.bytes,a.created_at,
                (SELECT COUNT(*) FROM documents d WHERE d.asset_id=a.id),
                (SELECT COUNT(*) FROM chunks c WHERE c.asset_id=a.id), a.original IS NOT NULL
                FROM assets a ORDER BY a.created_at,a.id""").fetchall()
        result = []
        for row in rows:
            item = dict(zip(("document_id","filename","bytes","created_at","pages","chunks","original_available"),row))
            item["original_available"] = bool(item["original_available"])
            result.append(item)
        return result

    def get_original(self, asset_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT filename,original FROM assets WHERE id=?", (asset_id,)).fetchone()
        if row is None or row[1] is None:
            return None
        return {"filename":row[0], "content":bytes(row[1])}

    def _chunk_rows(self, asset_ids: Sequence[str] | None = None) -> list[tuple]:
        if isinstance(asset_ids, (str,bytes)):
            raise ValueError("asset_ids must be a sequence of document IDs, not a string")
        if asset_ids is None:
            where,parameters = "", ()
        else:
            parameters = tuple(dict.fromkeys(asset_ids))
            if not parameters:
                return []
            where = " WHERE asset_id IN (" + ",".join("?" for _ in parameters) + ")"
        with self.connect() as db:
            return db.execute("SELECT id,asset_id,payload FROM chunks" + where + " ORDER BY id,asset_id", parameters).fetchall()

    def chunks(self, asset_ids: Sequence[str] | None = None) -> tuple[Chunk,...]:
        rows = self._chunk_rows(asset_ids)
        # 相同内容不同文件名可能共享ID；固定取第一份来源，防止重复投票。
        unique = {}
        for row in rows:
            chunk = Chunk(**json.loads(row[2]))
            unique.setdefault(chunk.chunk_id, chunk)
        return tuple(unique.values())

    def chunk_assets(self, asset_ids: Sequence[str] | None = None) -> dict[str,str]:
        """引用对应的上传ID，与chunks()使用相同的去重顺序和检索范围。"""
        assets = {}
        for chunk_id,asset_id,_ in self._chunk_rows(asset_ids):
            assets.setdefault(chunk_id,asset_id)
        return assets

    def delete(self, asset_id: str) -> bool:
        with self.connect() as db:
            return db.execute("DELETE FROM assets WHERE id=?", (asset_id,)).rowcount > 0


class CachedEmbeddingModel:
    """缓存以文本内容和模型指纹寻址，重启/增量上传复用已有向量。"""

    def __init__(self, model: object, store: DocumentStore, fingerprint: str) -> None:
        self.model, self.store, self.fingerprint = model, store, fingerprint

    def embed_passages(self, texts: Sequence[str]) -> tuple[tuple[float,...],...]:
        keys = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
        cached = {}
        with self.store.connect() as db:
            for key in dict.fromkeys(keys):
                row = db.execute("SELECT payload FROM vectors WHERE model=? AND text_hash=?",
                                 (self.fingerprint,key)).fetchone()
                if row:
                    cached[key] = tuple(json.loads(row[0]))
        missing = {key:text for key,text in zip(keys,texts) if key not in cached}
        if missing:
            vectors = self.model.embed_passages(tuple(missing.values()))
            # 不把失败/零向量永久存入缓存；否则一次模型异常会污染后续重启。
            _validate_embedding_batch(vectors, len(missing))
            with self.store.connect() as db:
                for key,vector in zip(missing,vectors,strict=True):
                    cached[key] = tuple(vector)
                    db.execute("INSERT OR REPLACE INTO vectors VALUES (?,?,?)",
                               (self.fingerprint,key,json.dumps(vector)))
        return tuple(cached[key] for key in keys)

    def embed_query(self, text: str) -> tuple[float,...]:
        return self.model.embed_query(text)
