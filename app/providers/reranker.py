"""Reranker Provider：两阶段检索的第二阶段（召回粗排 → 精排）。

- LocalCrossEncoderReranker: 本地 sentence-transformers CrossEncoder（bge-reranker-base）
- SiliconFlowReranker: SiliconFlow API（bge-reranker-v2-m3）
- NoReranker: 直通（关闭精排时使用）
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import httpx

from app.config import resolve_model_path, settings


class RerankerProvider(ABC):
    name: str = "base"

    @abstractmethod
    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """返回每个候选与 query 的相关性分数（降序排布由调用方决定）。"""


class LocalCrossEncoderReranker(RerankerProvider):
    name = "local:bge-reranker-base"

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.name = f"local:{Path(model_name).name}"
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(resolve_model_path(self.model_name), max_length=512, device="cpu")
        return self._model

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        import torch

        model = self._get_model()
        scores = model.predict([(query, t) for t in texts])
        # bge-reranker 输出原始 logits，过 sigmoid 得到 0~1 相关性
        return torch.sigmoid(torch.tensor(scores)).flatten().tolist()


class SiliconFlowReranker(RerankerProvider):
    name = "siliconflow:bge-reranker-v2-m3"

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        resp = httpx.post(
            f"{settings.siliconflow_base_url}/rerank",
            headers={"Authorization": f"Bearer {settings.siliconflow_api_key}"},
            json={
                "model": "BAAI/bge-reranker-v2-m3",
                "query": query,
                "documents": texts,
            },
            timeout=60,
        )
        resp.raise_for_status()
        results = sorted(resp.json()["results"], key=lambda r: r["index"])
        return [r["relevance_score"] for r in results]


class NoReranker(RerankerProvider):
    name = "none"

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        return [1.0 / (i + 1) for i in range(len(texts))]


def get_reranker_provider() -> RerankerProvider:
    kind = settings.rerank_provider.lower()
    if kind == "none":
        return NoReranker()
    if kind == "local":
        return LocalCrossEncoderReranker(settings.local_reranker_model)
    if kind == "siliconflow":
        if not settings.siliconflow_api_key:
            raise RuntimeError("RERANK_PROVIDER=siliconflow 需要在 .env 中配置 SILICONFLOW_API_KEY")
        return SiliconFlowReranker()
    raise ValueError(f"未知的 RERANK_PROVIDER: {settings.rerank_provider}")
