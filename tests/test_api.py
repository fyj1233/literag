"""API 集成测试：认证 + 权限 + 上传/检索/流式问答（mock provider，全离线）。"""
import json

import pytest

from app import auth


@pytest.fixture()
def client(rag_env):
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


@pytest.fixture()
def boss_headers(client):
    """用内置 boss 账号登录换取 token。"""
    resp = client.post("/api/auth/login",
                       json={"username": "boss", "password": "boss123"})
    token = resp.json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_health_open(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_protected_requires_token(client):
    assert client.get("/api/stats").status_code == 401
    assert client.post("/api/chat", json={"question": "测试"}).status_code == 401
    assert client.get("/api/documents").status_code == 401


def test_login_wrong_password(client):
    resp = client.post("/api/auth/login",
                       json={"username": "boss", "password": "nope"})
    assert resp.status_code == 401


def test_upload_and_list(rag_env, client, boss_headers):
    with client as c:
        content = "# 手册\n## 参数\n功率是三十八瓦，噪音很低。".encode("utf-8")
        resp = c.post("/api/documents", files={"file": ("手册.md", content, "text/markdown")},
                      headers=boss_headers)
        assert resp.status_code == 200
        assert resp.json()["chunks"] >= 1

        docs = c.get("/api/documents", headers=boss_headers).json()["documents"]
        entry = next(d for d in docs if d["source"] == "手册.md")
        assert entry["chunks"] >= 1 and entry["visibility"] == "L1"

        bad = c.post("/api/documents",
                     files={"file": ("virus.exe", b"xx", "application/octet-stream")},
                     headers=boss_headers)
        assert bad.status_code == 400


def test_upload_confidential_invisible_to_employee(rag_env, client, boss_headers):
    """核心权限测试：L4 文档对 employee 完全不可见（列表 + 检索 + 问答）。"""
    with client as c:
        c.post("/api/documents",
               files={"file": ("财务季报.md", "# 财报\n## 营收\n第三季度营收1.2亿元。".encode("utf-8"),
                               "text/markdown")},
               data={"visibility": "L4"},
               headers=boss_headers)

        emp_tok = c.post("/api/auth/login",
                         json={"username": "employee", "password": "employee123"}).json()["token"]
        emp = {"Authorization": f"Bearer {emp_tok}"}

        # 文档列表：employee 看不到 L4
        names = [d["source"] for d in c.get("/api/documents", headers=emp).json()["documents"]]
        assert "财务季报.md" not in names
        # 检索：employee 的结果里没有 L4
        hits = c.post("/api/search", json={"question": "营收多少", "mode": "hybrid"},
                      headers=emp).json()["hits"]
        assert all(h["source"] != "财务季报.md" for h in hits)
        # boss 一切正常
        boss_hits = c.post("/api/search", json={"question": "营收多少", "mode": "hybrid"},
                           headers=boss_headers).json()["hits"]
        assert boss_hits and any(h["source"] == "财务季报.md" for h in boss_hits)


def test_guest_cannot_upload(client, rag_env):
    guest_tok = client.post("/api/auth/login",
                            json={"username": "guest", "password": "guest123"}).json()["token"]
    resp = client.post("/api/documents",
                       files={"file": ("a.md", b"# t", "text/markdown")},
                       data={"visibility": "L1"},
                       headers={"Authorization": f"Bearer {guest_tok}"})
    assert resp.status_code == 403


def test_search_endpoint(rag_env, client, boss_headers):
    with client as c:
        content = "# FAQ\n## 功率\n额定功率为38瓦。".encode("utf-8")
        c.post("/api/documents", files={"file": ("faq.md", content, "text/markdown")},
               headers=boss_headers)
        resp = c.post("/api/search", json={"question": "功率多少", "mode": "hybrid"},
                      headers=boss_headers)
        assert resp.status_code == 200
        hits = resp.json()["hits"]
        assert hits and hits[0]["source"] == "faq.md"


def test_chat_stream_sse(rag_env, client, boss_headers):
    with client as c:
        content = "# 指南\n## 年假\n入职满一年有五天带薪年假。".encode("utf-8")
        c.post("/api/documents", files={"file": ("guide.md", content, "text/markdown")},
               headers=boss_headers)

        with c.stream("POST", "/api/chat", json={"question": "年假几天？", "mode": "full"},
                      headers=boss_headers) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            types = []
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    types.append(json.loads(line[6:])["type"])
            assert types[0] == "meta"
            assert "delta" in types
            assert types[-1] == "done"


def test_chat_mode_validation(rag_env, client, boss_headers):
    with client as c:
        resp = c.post("/api/chat", json={"question": "测试", "mode": "bad-mode"},
                      headers=boss_headers)
        assert resp.status_code == 400
        assert "mode" in resp.json()["detail"]


def test_chat_on_empty_kb(rag_env, client, boss_headers):
    with client as c:
        resp = c.post("/api/chat", json={"question": "测试", "mode": "naive"},
                      headers=boss_headers)
        assert resp.status_code == 409


def test_register_boss_only_and_audit(rag_env, client, boss_headers, monkeypatch, tmp_path):
    monkeypatch.setattr(auth, "DB_PATH", tmp_path / "u.db")   # 用户库隔离，避免真实库污染
    from app import auth as auth_mod
    auth_mod.seed()
    with client as c:
        resp = c.post("/api/auth/register",
                      json={"username": "newbie", "password": "pw123456", "role": "employee"},
                      headers=boss_headers)
        assert resp.status_code == 200
        # 员工不能创建账号
        emp_tok = c.post("/api/auth/login",
                         json={"username": "employee", "password": "employee123"}).json()["token"]
        assert c.post("/api/auth/register",
                      json={"username": "x2", "password": "pw123456", "role": "employee"},
                      headers={"Authorization": f"Bearer {emp_tok}"}).status_code == 403
        # boss 可看审计
        audit = c.get("/api/audit", headers=boss_headers)
        assert audit.status_code == 200 and isinstance(audit.json()["entries"], list)
