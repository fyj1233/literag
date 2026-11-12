"""CRAG-lite：精排置信分过低时触发纠正重查。"""
from app.config import settings
from app.providers.reranker import RerankerProvider


class _StubReranker(RerankerProvider):
    name = "stub"

    def __init__(self, score: float):
        self.score = score

    def rerank(self, query, texts):
        return [self.score] * len(texts)


def _ingest(rag_env, tmp_path):
    doc = tmp_path / "手册.md"
    doc.write_text("# 手册\n## 保修\n整机保修两年，电机保修五年。\n## 功率\n额定功率 38W。\n", encoding="utf-8")
    rag_env.ingest_file(doc)


def test_low_confidence_triggers_corrective(rag_env, tmp_path, monkeypatch):
    _ingest(rag_env, tmp_path)
    monkeypatch.setattr(settings, "rerank_provider", "local")
    monkeypatch.setattr(settings, "enable_corrective", True)
    monkeypatch.setattr(settings, "corrective_threshold", 0.5)  # 故意设高，必触发
    rag_env.retriever._reranker = _StubReranker(0.1)            # 注入低分精排

    result = rag_env.ask("保修多久？", mode="full")
    assert result.trace.corrective is True


def test_high_confidence_skips_corrective(rag_env, tmp_path, monkeypatch):
    _ingest(rag_env, tmp_path)
    monkeypatch.setattr(settings, "rerank_provider", "local")
    monkeypatch.setattr(settings, "enable_corrective", True)
    monkeypatch.setattr(settings, "corrective_threshold", 0.5)
    rag_env.retriever._reranker = _StubReranker(0.9)            # 高分不触发

    result = rag_env.ask("保修多久？", mode="full")
    assert result.trace.corrective is False


def test_disabled_or_non_full_mode_never_triggers(rag_env, tmp_path, monkeypatch):
    _ingest(rag_env, tmp_path)
    monkeypatch.setattr(settings, "rerank_provider", "local")
    monkeypatch.setattr(settings, "enable_corrective", False)
    rag_env.retriever._reranker = _StubReranker(0.0)
    assert rag_env.ask("保修多久？", mode="full").trace.corrective is False

    monkeypatch.setattr(settings, "enable_corrective", True)
    assert rag_env.ask("保修多久？", mode="naive").trace.corrective is False
