"""pytest 公共夹具：全部走 mock/hash provider，不下载模型、不需要 key、不碰真实索引。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def rag_env(tmp_path):
    """隔离环境：临时数据目录 + 轻量 provider（hash 向量 / mock LLM / 关闭精排）。"""
    from app.config import settings

    settings.data_dir = tmp_path / "data"
    settings.upload_dir = tmp_path / "uploads"
    settings.embedding_provider = "hash"
    settings.llm_provider = "mock"
    settings.rerank_provider = "none"
    settings.ensure_dirs()

    from app.rag import pipeline as pipeline_mod

    pipeline_mod._pipeline = None  # 重置进程级单例，强制按当前配置重建
    yield pipeline_mod.RAGPipeline()
    pipeline_mod._pipeline = None
