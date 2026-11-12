"""Pipeline 编排：ingest（文档 → 索引） 与 ask（问题 → 引用答案）两条主链路。"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings
from app.providers.embedding import get_embedding_provider
from app.providers.llm import get_llm_provider
from app.rag.chunking import Chunk, chunk_document
from app.rag.generation import align_citations, build_messages
from app.rag.loaders import load_file
from app.rag.query import hyde_document, optimize_query
from app.rag.retrieval import RetrievalTrace, Retriever
from app.rag.storage import BM25Store, Hit, ParentStore, VectorStore


@dataclass
class AskResult:
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    trace: RetrievalTrace = field(default_factory=RetrievalTrace)
    hits: list[Hit] = field(default_factory=list)


IMAGE_SUFFIX = {".png", ".jpg", ".jpeg", ".webp"}   # 可直接上传入库的图片格式


class RAGPipeline:
    def __init__(self):
        self.embedding = get_embedding_provider()
        self.llm = get_llm_provider()
        self.vector_store = VectorStore(self.embedding.name)
        self.bm25_store = BM25Store()
        self.parent_store = ParentStore()
        self.retriever = Retriever(self.vector_store, self.bm25_store)

    # ==================== 摄入 ====================
    @staticmethod
    def _chunk_id(source: str, chunk_index: int) -> str:
        raw = f"{source}::{chunk_index}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def ingest_file(self, file_path: str | Path, source_name: str | None = None,
                    visibility: str = "L1") -> dict:
        """单文件入库入口：图片走识图分支，其余走"解析→切分"分支。"""
        path = Path(file_path)
        source = source_name or path.name
        if path.suffix.lower() in IMAGE_SUFFIX:
            return self._ingest_image_file(path, source, visibility)
        return self._ingest_document(path, source, visibility)

    def _ingest_image_file(self, path: Path, source: str, visibility: str) -> dict:
        """图片文件直接入库：文件本身交给识图模型，描述作为唯一 chunk（继承密级）。"""
        t0 = time.perf_counter()
        self.delete_source(source)   # 同名重传幂等
        from app.rag.vision import describe_image, to_png_bytes

        png = to_png_bytes(path.read_bytes())   # 非法图片在此抛异常
        desc = describe_image(png) if settings.vision_enabled else None
        if not desc:
            return {"source": source, "chunks": 0, "images": 0,
                    "elapsed": round(time.perf_counter() - t0, 2),
                    "note": "识图未返回有效描述，未入库（请检查识图模型配置）"}
        title = Path(source).stem
        chunk = Chunk(
            text=desc,
            embedding_text=f"{title} > 图片\n{desc}",
            metadata={"source": source, "title": title, "section_path": "图片",
                      "chunk_index": 0, "visibility": visibility},
        )
        self.vector_store.upsert(
            ids=[self._chunk_id(source, 0)],
            embeddings=self.embedding.embed([chunk.embedding_text]),
            texts=[chunk.text],
            metadatas=[chunk.metadata],
        )
        self.parent_store.delete_source(source)
        self._rebuild_bm25()
        return {"source": source, "chunks": 1, "images": 1,
                "elapsed": round(time.perf_counter() - t0, 2)}

    def _ingest_document(self, path: Path, source: str, visibility: str) -> dict:
        """解析单文件 → 切分 → 向量化 → 双索引写入（visibility 为文档密级）。"""
        t0 = time.perf_counter()
        # 同名文档重传：先删旧索引，保证幂等
        self.delete_source(source)

        chunks = []      # [(Chunk, parent_id)]
        parents: dict[str, dict] = {}   # parent_id -> 章节级全文信息
        for sec in load_file(path):
            pid = hashlib.md5(f"{source}::{sec.section_path}::{sec.page or 0}".encode("utf-8")).hexdigest()
            parents[pid] = {
                "parent_id": pid, "source": source, "title": sec.title,
                "section_path": sec.section_path, "page": sec.page, "text": sec.text,
            }
            for c in chunk_document(
                text=sec.text,
                title=sec.title,
                section_path=sec.section_path,
                page=sec.page,
                source=source,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            ):
                chunks.append((c, pid))

        # 图片理解（可开关）：抽取内嵌图片 → 识图模型生成中文描述 → 特殊 chunk 入库。
        # 失败/超限/未启用都优雅降级，不阻断摄入主链路。整份文档只执行一次。
        n_images = 0
        if settings.vision_enabled:
            from app.rag.vision import describe_image, extract_images

            for img_i, img in enumerate(extract_images(path)[: max(settings.vision_max_images, 0)]):
                desc = describe_image(img["png_bytes"])
                if not desc:
                    continue
                page = img.get("page")
                sec_path = f"图片(第{page}页)" if page else f"图片{img_i + 1}"
                title = Path(source).stem
                chunks.append((
                    Chunk(
                        text=desc,
                        embedding_text=f"{title} > {sec_path}\n{desc}",
                        metadata={"source": source, "title": title, "section_path": sec_path,
                                  "page": page},
                    ),
                    None,   # 图片描述没有章节父块
                ))
                n_images += 1

        if not chunks:
            return {"source": source, "chunks": 0, "elapsed": 0.0}

        # 各章节独立切分时 chunk_index 会重复，入库前统一重编号保证全局唯一
        for i, (c, pid) in enumerate(chunks):
            c.metadata["chunk_index"] = i
            c.metadata["parent_id"] = pid
            c.metadata["visibility"] = visibility
        ids = [self._chunk_id(source, c.metadata["chunk_index"]) for c, _ in chunks]
        embeddings = self.embedding.embed([c.embedding_text for c, _ in chunks])
        self.vector_store.upsert(
            ids=ids,
            embeddings=embeddings,
            texts=[c.text for c, _ in chunks],
            metadatas=[c.metadata for c, _ in chunks],
        )
        self.parent_store.set_source(source, parents)
        self._rebuild_bm25()

        return {
            "source": source,
            "chunks": len(chunks),
            "images": n_images,
            "elapsed": round(time.perf_counter() - t0, 2),
        }

    def ingest_directory(self, dir_path: str | Path, visibility: str = "L1") -> list[dict]:
        results = []
        for path in sorted(Path(dir_path).iterdir()):
            if path.suffix.lower() in (".md", ".markdown", ".txt", ".pdf", ".docx"):
                results.append(self.ingest_file(path, visibility=visibility))
        return results

    def delete_source(self, source: str) -> int:
        deleted = self.vector_store.delete_source(source)
        self.parent_store.delete_source(source)
        if deleted:
            self._rebuild_bm25()
        return deleted

    def set_source_visibility(self, source: str, visibility: str) -> int:
        """运行时调整已入库文档的密级：原位更新 Chroma 元数据（无需重灌索引）。

        BM25 路径的密级过滤发生在"从向量库补全元数据"之后，会自动跟随本次更新。
        返回受影响的 chunk 数；文档不存在抛 ValueError。
        """
        got = self.vector_store.collection.get(where={"source": source}, include=["metadatas"])
        if not got["ids"]:
            raise ValueError(f"未找到文档: {source}")
        metas = []
        for meta in got["metadatas"]:
            meta = dict(meta or {})
            meta["visibility"] = visibility
            metas.append(meta)
        self.vector_store.collection.update(ids=got["ids"], metadatas=metas)
        return len(got["ids"])

    def _rebuild_bm25(self) -> None:
        got = self.vector_store.collection.get(include=["documents"])
        self.bm25_store.rebuild(list(zip(got["ids"], got["documents"])))

    def stats(self) -> dict:
        return {
            "chunks": self.vector_store.count(),
            "sources": self.vector_store.sources(),
            "embedding": self.embedding.name,
            "reranker": settings.rerank_provider,
            "llm_provider": settings.llm_provider,
        }

    # ==================== 问答 ====================
    def _maybe_corrective(self, oq, hits: list[Hit], mode: str, top_k: int,
                          allowed: list[str] | None = None) -> tuple[list[Hit], bool]:
        """CRAG-lite：精排 Top1 置信分低于阈值时，扩大候选集纠正重查一次，取两次较优。

        仅在 mode=full 且启用精排时生效；只做一次纠正，控制延迟。
        """
        if not (settings.enable_corrective and mode == "full"):
            return hits, False
        if settings.rerank_provider == "none" or not hits:
            return hits, False
        if hits[0].score >= settings.corrective_threshold:
            return hits, False
        queries = list(dict.fromkeys(oq.recall_queries + [oq.original]))
        hits2, _ = self.retriever.retrieve(
            queries, mode=mode, top_k=top_k, candidate_k=settings.candidate_k * 2,
            allowed=allowed
        )
        if hits2 and hits2[0].score > hits[0].score:
            return hits2, True
        return hits, True

    def _context_blocks(self, hits: list[Hit]) -> list[Hit]:
        """Parent-Child：开启时把同章节的命中子块合并为父块（全文截断）作为生成上下文。

        检索仍以小块为主保证精度；合并按首次命中顺序、保留组内最高分。
        """
        if not (settings.enable_parent_child and hits):
            return hits
        blocks: list[Hit] = []
        by_pid: dict[str, Hit] = {}
        for h in hits:
            if not h.parent_id:
                blocks.append(h)
                continue
            if h.parent_id in by_pid:
                by_pid[h.parent_id].score = max(by_pid[h.parent_id].score, h.score)
                continue
            info = self.parent_store.get(h.parent_id)
            if not info:
                blocks.append(h)
                continue
            block = Hit(
                chunk_id=h.parent_id,
                text=info["text"][: settings.parent_max_chars],
                embedding_text=info["text"],
                score=h.score,
                source=info["source"],
                title=info["title"],
                section_path=info["section_path"],
                chunk_index=0,
                page=info.get("page"),
                parent_id=h.parent_id,
            )
            by_pid[h.parent_id] = block
            blocks.append(block)
        return blocks

    def ask(self, question: str, mode: str = "full", history: list[dict] | None = None,
            top_k: int | None = None, allowed: list[str] | None = None,
            image_data_url: str | None = None) -> AskResult:
        """非流式：一次性返回完整结果（评估脚本/测试用）。"""
        result = AskResult()
        oq = optimize_query(question, self.llm, enable=mode == "full", history=history)
        if oq.rewritten and oq.rewritten != question:
            result.trace.rewritten = oq.rewritten
        result.trace.sub_queries = list(oq.sub_queries)

        if settings.enable_hyde and mode == "full":
            oq.hyde_doc = hyde_document(question, self.llm)

        hits, rtrace = self.retriever.retrieve(oq.recall_queries, mode=mode, top_k=top_k,
                                               allowed=allowed)
        hits, corrected = self._maybe_corrective(oq, hits, mode, top_k or settings.final_top_k,
                                                allowed=allowed)
        hits = self._context_blocks(hits)
        result.hits = hits
        result.trace.queries = rtrace.queries
        result.trace.vector_hits = rtrace.vector_hits
        result.trace.bm25_hits = rtrace.bm25_hits
        result.trace.fused_candidates = rtrace.fused_candidates
        result.trace.timings = rtrace.timings
        result.trace.corrective = corrected

        messages = build_messages(question, hits, history, image_data_url=image_data_url)
        result.answer = self.llm.chat(messages)
        result.citations = align_citations(result.answer, hits)
        return result

    def ask_stream(
        self, question: str, mode: str = "full", history: list[dict] | None = None,
        top_k: int | None = None, allowed: list[str] | None = None,
        image_data_url: str | None = None,
    ) -> Iterator[dict]:
        """流式：先发 meta（检索轨迹），再逐段发 delta，最后发 citations。"""
        t0 = time.perf_counter()
        oq = optimize_query(question, self.llm, enable=mode == "full", history=history)
        if settings.enable_hyde and mode == "full":
            oq.hyde_doc = hyde_document(question, self.llm)

        hits, rtrace = self.retriever.retrieve(oq.recall_queries, mode=mode, top_k=top_k,
                                               allowed=allowed)
        hits, corrected = self._maybe_corrective(oq, hits, mode, top_k or settings.final_top_k,
                                                allowed=allowed)
        hits = self._context_blocks(hits)
        meta = {
            "type": "meta",
            "rewritten": oq.rewritten if oq.rewritten != question else "",
            "sub_queries": oq.sub_queries,
            "queries": rtrace.queries,
            "vector_hits": rtrace.vector_hits,
            "bm25_hits": rtrace.bm25_hits,
            "fused_candidates": rtrace.fused_candidates,
            "retrieval_ms": round((time.perf_counter() - t0) * 1000),
            "hits": [h.to_dict() for h in hits],
            "mode": mode,
            "corrective": corrected,
        }
        yield meta

        messages = build_messages(question, hits, history, image_data_url=image_data_url)
        if image_data_url and hasattr(self.llm, "max_tokens"):
            # 贴图问答：识图推理消耗额外 token，给生成预算留足余量
            self.llm.max_tokens = max(self.llm.max_tokens or 0, 4000)
        answer_parts: list[str] = []
        for delta in self.llm.chat_stream(messages):
            answer_parts.append(delta)
            yield {"type": "delta", "text": delta}

        answer = "".join(answer_parts)
        yield {
            "type": "done",
            "answer": answer,
            "citations": align_citations(answer, hits),
            "total_ms": round((time.perf_counter() - t0) * 1000),
        }


def get_pipeline() -> RAGPipeline:
    """进程级单例：避免每次请求重建索引连接与模型。"""
    global _pipeline  # noqa: PLW0603
    if "_pipeline" not in globals() or _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
