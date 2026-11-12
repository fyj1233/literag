"""检索层：向量召回、BM25 稀疏召回、RRF 融合、CrossEncoder 精排。

RRF（Reciprocal Rank Fusion）：对每个候选，score = Σ 1/(k + rank_i)，
只看排名不看原始分数，天然规避向量余弦分与 BM25 分无法直接相加的问题。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from app.config import settings
from app.providers.embedding import get_embedding_provider
from app.providers.reranker import get_reranker_provider
from app.rag.storage import BM25Store, Hit, VectorStore


@dataclass
class RetrievalTrace:
    """一次检索的全链路轨迹：前端「检索过程」面板与评估脚本共用。"""

    queries: list[str] = field(default_factory=list)        # 实际用于召回的查询（含子查询）
    rewritten: str = ""                                     # LLM 改写后的主查询
    sub_queries: list[str] = field(default_factory=list)
    vector_hits: int = 0
    bm25_hits: int = 0
    fused_candidates: int = 0
    timings: dict[str, float] = field(default_factory=dict) # 各阶段耗时(秒)
    corrective: bool = False                                # 是否触发了 CRAG-lite 纠正重查


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """把多路召回的 id 排名融合为统一得分排名。"""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def extract_citations(answer: str) -> list[int]:
    """从答案中提取 [1][2] 引用编号（去重、升序）。"""
    nums = {int(n) for n in re.findall(r"\[(\d{1,2})\]", answer)}
    return sorted(nums)


class Retriever:
    """面向单次查询的多路检索与融合。"""

    def __init__(self, vector_store: VectorStore, bm25_store: BM25Store):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self._embedding = None
        self._reranker = None

    @property
    def embedding(self):
        if self._embedding is None:
            self._embedding = get_embedding_provider()
        return self._embedding

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = get_reranker_provider()
        return self._reranker

    # ---------- 单路召回 ----------
    def vector_search(self, query: str, top_k: int,
                      allowed: list[str] | None = None) -> list[Hit]:
        vec = self.embedding.embed([query])[0]
        return self.vector_store.query(vec, top_k, allowed=allowed)

    def bm25_search(self, query: str, top_k: int) -> list[Hit]:
        id_scores = self.bm25_store.search(query, top_k)
        hits: list[Hit] = []
        for cid, score in id_scores:
            doc = self.bm25_store.get_doc(cid)
            if doc is None:
                continue
            # BM25 只存了 embedding_text；正文从向量库取（一致以 Chroma 为准）
            hits.append(
                Hit(chunk_id=cid, text="", embedding_text=doc, score=score,
                    source="", title="", section_path="", chunk_index=0)
            )
        return hits

    def _fill_from_vector(self, hits: list[Hit]) -> list[Hit]:
        """BM25 命中的 chunk 缺正文/元数据，用向量库补全。"""
        missing = [h for h in hits if not h.source]
        if not missing:
            return hits
        got = self.vector_store.collection.get(
            ids=[h.chunk_id for h in missing], include=["documents", "metadatas"]
        )
        by_id = {cid: (doc, meta or {}) for cid, doc, meta in zip(
            got["ids"], got["documents"], got["metadatas"])}
        for h in hits:
            if h.chunk_id in by_id:
                doc, meta = by_id[h.chunk_id]
                h.text = doc
                h.source = meta.get("source", "")
                h.title = meta.get("title", "")
                h.section_path = meta.get("section_path", "")
                h.chunk_index = int(meta.get("chunk_index", 0))
                h.page = meta.get("page")
                h.visibility = meta.get("visibility", "L1")
                h.parent_id = meta.get("parent_id")
        return [h for h in hits if h.source]  # 向量库里也找不到的丢弃（索引不一致保护）

    # ---------- 融合检索 ----------
    def retrieve(
        self,
        queries: list[str],
        mode: str = "full",
        top_k: int | None = None,
        candidate_k: int | None = None,
        allowed: list[str] | None = None,
    ) -> tuple[list[Hit], RetrievalTrace]:
        """queries: 主查询 + 子查询列表；mode: naive / hybrid / rerank / full。

        allowed: 当前角色可见的密级列表（ACL）——在召回阶段过滤，
        无权限内容不进入融合/精排/Prompt 的任何环节（默认拒绝原则）。

        naive  : 仅向量召回（第一条 query）
        hybrid : 向量 + BM25，RRF 融合
        rerank : hybrid + CrossEncoder 精排
        full   : rerank + 多查询（每条 query 都参与召回）
        """
        trace = RetrievalTrace()
        top_k = top_k or settings.final_top_k
        t0 = time.perf_counter()

        main_query = queries[0]
        use_hybrid = settings.enable_hybrid and mode in ("hybrid", "rerank", "full")
        use_rerank = settings.enable_rerank and mode in ("rerank", "full")
        use_multi = settings.enable_multi_query and mode == "full" and len(queries) > 1
        recall_queries = [main_query] if not use_multi else queries
        trace.queries = list(recall_queries)

        vector_hits: list[Hit] = []
        bm25_hits: list[Hit] = []
        for q in recall_queries:
            vector_hits.extend(self.vector_search(q, settings.vector_top_k, allowed=allowed))
            if use_hybrid:
                bm25_hits.extend(self.bm25_search(q, settings.bm25_top_k))
        if allowed is not None:   # ACL 双保险：向量库 where 之外再做一次应用层过滤
            aset = set(allowed)
            vector_hits = [h for h in vector_hits if h.visibility in aset]
        trace.timings["recall"] = round(time.perf_counter() - t0, 3)
        trace.vector_hits = len(vector_hits)
        trace.bm25_hits = len(bm25_hits)

        if not use_hybrid:
            fused = vector_hits
        else:
            # 多路多查询结果先去重（同一 chunk 保留最高分出现）
            def dedup(hits: list[Hit]) -> list[Hit]:
                best: dict[str, Hit] = {}
                for h in hits:
                    if h.chunk_id not in best or h.score > best[h.chunk_id].score:
                        best[h.chunk_id] = h
                return list(best.values())

            v_hits, b_hits = dedup(vector_hits), dedup(bm25_hits)
            b_hits = self._fill_from_vector(b_hits)
            if allowed is not None:
                # ★ ACL 过滤必须在元数据补全之后：BM25 候选的密级在补全前是未知的
                aset = set(allowed)
                b_hits = [h for h in b_hits if h.visibility in aset]
            rankings = [
                [h.chunk_id for h in sorted(v_hits, key=lambda x: -x.score)],
                [h.chunk_id for h in sorted(b_hits, key=lambda x: -x.score)],
            ]
            t1 = time.perf_counter()
            fused_ids = rrf_fuse(rankings, k=settings.rrf_k)
            trace.timings["fuse_rrf"] = round(time.perf_counter() - t1, 3)
            trace.fused_candidates = len(fused_ids)
            by_id = {h.chunk_id: h for h in v_hits + b_hits}
            fused = []
            for cid, score in fused_ids[: candidate_k or settings.candidate_k]:
                h = by_id[cid]
                h.score = score  # RRF 分
                fused.append(h)

        # ---------- 精排 ----------
        if use_rerank and fused:
            t2 = time.perf_counter()
            texts = [f"{h.title} > {h.section_path}\n{h.text}" if h.section_path else h.text
                     for h in fused]
            scores = self.reranker.rerank(main_query, texts)
            for h, s in zip(fused, scores):
                h.score = s
            fused = sorted(fused, key=lambda x: -x.score)
            trace.timings["rerank"] = round(time.perf_counter() - t2, 3)

        # naive/hybrid 模式下按分数截断
        fused = sorted(fused, key=lambda x: -x.score)[:top_k]
        return fused, trace
