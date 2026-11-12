"""设置面板 API 测试：boss 专属、key 掩码、热更新、.env 持久化隔离。"""
import pytest

from app import auth, config
from app.config import settings

SNAPSHOT_FIELDS = [
    "llm_provider", "llm_base_url", "llm_api_key", "llm_model", "llm_temperature", "llm_max_tokens",
    "enable_hybrid", "enable_rerank", "enable_query_rewrite", "enable_multi_query",
    "enable_hyde", "enable_corrective", "enable_parent_child",
    "vector_top_k", "bm25_top_k", "rrf_k", "candidate_k", "final_top_k",
    "chunk_size", "chunk_overlap",
]


@pytest.fixture()
def boss_client(rag_env, monkeypatch, tmp_path):
    """boss 登录 + .env/用户库隔离 + 全局配置快照恢复。"""
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setattr(config, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(auth, "DB_PATH", tmp_path / "u.db")
    auth.seed()
    client = TestClient(app)
    tok = client.post("/api/auth/login", json={"username": "boss", "password": "boss123"}).json()["token"]
    client.headers.update({"Authorization": f"Bearer {tok}"})

    snap = {name: getattr(settings, name) for name in SNAPSHOT_FIELDS}
    yield client
    for name, value in snap.items():
        setattr(settings, name, value)


def test_employee_forbidden(rag_env, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setattr(config, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(auth, "DB_PATH", tmp_path / "u.db")
    auth.seed()
    client = TestClient(app)
    tok = client.post("/api/auth/login", json={"username": "employee", "password": "employee123"}).json()["token"]
    resp = client.get("/api/settings", headers={"Authorization": f"Bearer {tok}"})
    assert resp.status_code == 403


def test_get_masks_api_key(boss_client):
    settings.llm_api_key = "sk-verylongsecretkey-1234567890"
    resp = boss_client.get("/api/settings")
    assert resp.status_code == 200
    body = resp.json()
    assert "****" in body["llm"]["api_key_masked"]
    assert body["llm"]["has_key"] is True
    assert "sk-verylongsecretkey" not in resp.text, "完整 key 绝不能出现在响应里"


def test_put_llm_hot_updates_provider(boss_client):
    resp = boss_client.put("/api/settings/llm", json={
        "base_url": "https://example.com/v1", "model": "test-model",
        "temperature": 0.5, "max_tokens": 512})   # api_key 留空 = 保持现有
    assert resp.status_code == 200
    assert resp.json()["base_url"] == "https://example.com/v1"
    assert settings.llm_model == "test-model"
    assert settings.llm_provider == "openai"
    from app.rag.pipeline import get_pipeline

    assert get_pipeline().llm.model == "test-model", "Provider 应已热重建"


def test_put_llm_empty_key_keeps_existing(boss_client):
    settings.llm_api_key = "sk-keep-me-1234567890"
    boss_client.put("/api/settings/llm", json={
        "base_url": "https://example.com/v1", "model": "m", "temperature": 0.3, "max_tokens": 512})
    assert settings.llm_api_key == "sk-keep-me-1234567890", "留空 key 应保持原值"


def test_put_features(boss_client):
    resp = boss_client.put("/api/settings/features", json={"enable_hyde": True, "enable_hybrid": False})
    assert resp.status_code == 200
    assert settings.enable_hyde is True and settings.enable_hybrid is False
    bad = boss_client.put("/api/settings/features", json={"enable_hack": True})
    assert bad.status_code == 400


def test_put_retrieval_validation(boss_client):
    assert boss_client.put("/api/settings/retrieval", json={"vector_top_k": 999}).status_code == 400
    assert boss_client.put("/api/settings/retrieval", json={"rrf_k": 80}).status_code == 200
    assert settings.rrf_k == 80


def test_persist_env_roundtrip(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# 注释必须保留\nLLM_MODEL=old\nCHUNK_SIZE=500\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_PATH", env)
    config.persist_env({"LLM_MODEL": "new-model", "NEW_KEY": "abc"})
    text = env.read_text(encoding="utf-8")
    assert "# 注释必须保留" in text
    assert "LLM_MODEL=new-model" in text and "NEW_KEY=abc" in text
    assert "LLM_MODEL=old" not in text
