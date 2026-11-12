"""Embedding Provider 抽象与三种实现：

- LocalBGEMiddleware: 本地 sentence-transformers（bge-small-zh-v1.5，无需 key）
- SiliconFlowEmbedding: SiliconFlow API（BAAI/bge-m3）
- HashEmbedding: 确定性哈希向量，仅用于单测/离线调试（零依赖、零下载）

关键设计：不同模型的向量维度不同，所以向量库 collection 以「模型名」命名，
切换 Provider 后重新 ingest 即可，互不污染。
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

import httpx

from app.config import resolve_model_path, settings


class EmbeddingProvider(ABC):
    name: str = "base"

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量向量化。"""


class LocalBGEMiddleware(EmbeddingProvider):
    """本地 BGE 模型；首次调用时才加载（懒加载），便于测试环境跳过。"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.name = f"local:{Path(model_name).name}"
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(resolve_model_path(self.model_name), device="cpu")
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        vecs = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32
        )
        return [v.tolist() for v in vecs]


class SiliconFlowEmbedding(EmbeddingProvider):
    name = "siliconflow:bge-m3"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = httpx.post(
            f"{settings.siliconflow_base_url}/embeddings",
            headers={"Authorization": f"Bearer {settings.siliconflow_api_key}"},
            json={"model": "BAAI/bge-m3", "input": texts},
            timeout=60,
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]


class HashEmbedding(EmbeddingProvider):
    """确定性哈希向量：不依赖任何模型/网络，供单测与离线冒烟使用。"""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = f"hash:{dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        import jieba

        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in jieba.lcut(text):
                h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out


def get_embedding_provider() -> EmbeddingProvider:
    kind = settings.embedding_provider.lower()
    if kind == "local":
        return LocalBGEMiddleware(settings.local_embedding_model)
    if kind == "siliconflow":
        if not settings.siliconflow_api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=siliconflow 需要在 .env 中配置 SILICONFLOW_API_KEY")
        return SiliconFlowEmbedding()
    if kind == "hash":
        return HashEmbedding()
    raise ValueError(f"未知的 EMBEDDING_PROVIDER: {settings.embedding_provider}")
