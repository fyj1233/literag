"""存储层：ChromaDB（向量，持久化） + rank_bm25（稀疏索引，pickle 持久化）。

设计要点：
1. collection 以 embedding 模型名命名 —— 不同向量模型维度不同，隔离存放；
2. chunk 正文存 Chroma documents，BM25 只存 id 与分词结果，正文以 Chroma 为准；
3. BM25 索引在任何写操作后整体重建（demo 规模下毫秒级，换取实现简单可靠）。
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb

from app.config import settings


@dataclass
class Hit:
    """检索命中的一个 chunk（开启 Parent-Child 时也可能是父块）。"""

    chunk_id: str
    text: str
    embedding_text: str
    score: float          # 统一为 0~1（越大越相关）
    source: str
    title: str
    section_path: str
    chunk_index: int
    page: int | None = None
    parent_id: str | None = None   # 所属章节父块的 id
    visibility: str = "L1"         # 文档密级（继承自文档，检索层 ACL 依据）

    def to_dict(self) -> dict:
        d = {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": round(self.score, 4),
            "source": self.source,
            "title": self.title,
            "section_path": self.section_path,
            "chunk_index": self.chunk_index,
            "visibility": self.visibility,
        }
        if self.page is not None:
            d["page"] = self.page
        if self.parent_id is not None:
            d["parent_id"] = self.parent_id
        return d


def _safe_collection_name(model_name: str) -> str:
    """Chroma collection 名只允许 [a-zA-Z0-9_-]{3,63}。"""
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", model_name).strip("_")
    return f"chunks_{name}"[:63]


class VectorStore:
    def __init__(self, embedding_model_name: str):
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.collection_name = _safe_collection_name(embedding_model_name)
        self.collection = client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    # ---- 写 ----
    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict],
    ) -> None:
        self.collection.upsert(
            ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas
        )

    def delete_source(self, source: str) -> int:
        before = self.collection.count()
        self.collection.delete(where={"source": source})
        return before - self.collection.count()

    def reset(self) -> None:
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        client.delete_collection(self.collection_name)
        self.collection = client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    # ---- 读 ----
    def count(self) -> int:
        return self.collection.count()

    def sources(self, allowed: list[str] | None = None) -> dict[str, int]:
        """各来源文档的 chunk 数；allowed 给定时只统计该角色可见密级的文档。"""
        include = ["metadatas"]
        where = {"visibility": {"$in": allowed}} if allowed else None
        got = (
            self.collection.get(include=include, where=where)
            if where
            else self.collection.get(include=include)
        )
        out: dict[str, int] = {}
        for meta in got["metadatas"]:
            src = (meta or {}).get("source", "unknown")
            out[src] = out.get(src, 0) + 1
        return dict(sorted(out.items()))

    def query(self, embedding: list[float], top_k: int,
              allowed: list[str] | None = None) -> list[Hit]:
        if self.collection.count() == 0:
            return []
        where = {"visibility": {"$in": allowed}} if allowed else None
        res = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
            where=where,
        )
        hits: list[Hit] = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            meta = meta or {}
            # cosine distance ∈ [0,2]，归一为 0~1 相似度
            score = max(0.0, 1.0 - dist / 2.0)
            hits.append(self._to_hit(cid, doc, meta, score))
        return hits

    @staticmethod
    def _to_hit(cid: str, doc: str, meta: dict, score: float) -> Hit:
        return Hit(
            chunk_id=cid,
            text=doc,
            embedding_text=doc,
            score=score,
            source=meta.get("source", ""),
            title=meta.get("title", ""),
            section_path=meta.get("section_path", ""),
            chunk_index=int(meta.get("chunk_index", 0)),
            page=meta.get("page"),
            parent_id=meta.get("parent_id"),
            visibility=meta.get("visibility", "L1"),
        )


class ParentStore:
    """Parent-Child 分块的父块存储：parent_id → 章节级全文。

    用法："小块检索、父块生成"——子块（chunk）负责精准命中，
    命中后取其所属章节全文（截断）交给 LLM，兼顾检索精度与上下文完整性。
    """

    def __init__(self):
        self.path: Path = settings.bm25_dir / "parents.pkl"
        self._data: dict[str, dict[str, dict]] = {}   # source -> {parent_id -> info}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(self._data, f)

    def _load(self) -> None:
        if self._data or not self.path.exists():
            return
        with open(self.path, "rb") as f:
            self._data = pickle.load(f)

    def set_source(self, source: str, parents: dict[str, dict]) -> None:
        """幂等写入某文档的全部父块（重传同文档时整体覆盖）。"""
        self._load()
        self._data[source] = parents
        self._save()

    def delete_source(self, source: str) -> None:
        self._load()
        if source in self._data:
            del self._data[source]
            self._save()

    def get(self, parent_id: str) -> dict | None:
        self._load()
        for parents in self._data.values():
            if parent_id in parents:
                return parents[parent_id]
        return None


class BM25Store:
    def __init__(self):
        self.path: Path = settings.bm25_dir / "bm25.pkl"
        self._ids: list[str] = []
        self._docs: list[str] = []       # 用于 bm25 的 embedding_text
        self._corpus_tokens: list[list[str]] = []
        self._bm25 = None

    # ---- 分词 ----
    @staticmethod
    def tokenize(text: str) -> list[str]:
        import jieba

        jieba.setLogLevel(60)
        return [t for t in jieba.lcut_for_search(text) if t.strip()]

    # ---- 写 ----
    def rebuild(self, entries: list[tuple[str, str]]) -> None:
        """entries = [(chunk_id, embedding_text)]，全量重建索引。"""
        from rank_bm25 import BM25Okapi

        self._ids = [e[0] for e in entries]
        self._docs = [e[1] for e in entries]
        self._corpus_tokens = [self.tokenize(d) for d in self._docs]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._ids else None
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump({"ids": self._ids, "docs": self._docs}, f)

    # ---- 读 ----
    def _ensure_loaded(self) -> None:
        if self._bm25 is None and self.path.exists():
            with open(self.path, "rb") as f:
                data = pickle.load(f)
            self._ids, self._docs = data["ids"], data["docs"]
            self._corpus_tokens = [self.tokenize(d) for d in self._docs]
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self._corpus_tokens) if self._ids else None

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """返回 [(chunk_id, 归一化分数)]。"""
        self._ensure_loaded()
        if self._bm25 is None or not self._ids:
            return []
        tokens = self.tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(self._ids, scores), key=lambda x: -x[1])[:top_k]
        max_score = ranked[0][1] if ranked and ranked[0][1] > 0 else 1.0
        return [(cid, max(0.0, s / max_score)) for cid, s in ranked if s > 0]

    def get_doc(self, chunk_id: str) -> str | None:
        self._ensure_loaded()
        if chunk_id in self._ids:
            return self._docs[self._ids.index(chunk_id)]
        return None
