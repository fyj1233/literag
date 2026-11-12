"""分级权限测试：检索层 ACL（员工/访客看不到机密）+ JWT 认证（签名/过期/登录）。"""
import pytest

from app import auth
from app.config import ROLE_LEVELS, settings


@pytest.fixture()
def acl_env(rag_env, tmp_path, monkeypatch):
    """两份文档：L1 公开 + L4 机密；关闭 Parent-Child 便于断言来源。"""
    monkeypatch.setattr(settings, "enable_parent_child", False)
    (tmp_path / "公开.md").write_text("# 公开\n## 制度\n公司班车线路每日共三条。\n", encoding="utf-8")
    (tmp_path / "机密.md").write_text("# 机密\n## 营收\n2026年第三季度公司营收1.2亿元，净利润1800万元。\n", encoding="utf-8")
    rag_env.ingest_file(tmp_path / "公开.md", visibility="L1")
    rag_env.ingest_file(tmp_path / "机密.md", visibility="L4")
    return rag_env


def allowed_for(role: str) -> list[str]:
    return ROLE_LEVELS[role]


# ---------- 检索层 ACL ----------
def test_employee_cannot_reach_confidential(acl_env):
    hits, _ = acl_env.retriever.retrieve(["营收多少"], mode="hybrid",
                                         allowed=allowed_for("employee"))
    assert all(h.source != "机密.md" for h in hits), "员工的检索结果里不允许出现机密文档"
    result = acl_env.ask("营收多少", mode="full", allowed=allowed_for("employee"))
    assert all(c["source"] != "机密.md" for c in result.citations)


def test_boss_can_reach_confidential(acl_env):
    result = acl_env.ask("第三季度营收是多少？", mode="full", allowed=allowed_for("boss"))
    assert any(c["source"] == "机密.md" for c in result.citations), "boss 应能检索到机密文档"


def test_guest_sees_only_public(acl_env):
    result = acl_env.ask("班车线路有几条？", mode="full", allowed=allowed_for("guest"))
    assert all(c["source"] != "机密.md" for c in result.citations)
    assert any(c["source"] == "公开.md" for c in result.citations)


def test_unknown_role_denied_by_default(acl_env):
    result = acl_env.ask("营收多少", mode="full", allowed=[])
    assert result.citations == [] or all(c["source"] != "机密.md" for c in result.citations)


# ---------- 认证（JWT / 密码） ----------
def test_login_and_wrong_password(monkeypatch, tmp_path):
    monkeypatch.setattr(auth, "DB_PATH", tmp_path / "u.db")
    auth.seed()
    ok = auth.login("boss", "boss123")
    assert ok and ok["role"] == "boss" and ok["token"].count(".") == 2
    assert auth.login("boss", "wrong-password") is None
    assert auth.login("no-such-user", "x") is None


def test_jwt_signature_and_expiry(monkeypatch, tmp_path):
    monkeypatch.setattr(auth, "DB_PATH", tmp_path / "u.db")
    auth.seed()
    token = auth.make_token("employee", "employee")
    claims = auth.verify_token(token)
    assert claims and claims["role"] == "employee" and claims["sub"] == "employee"
    assert auth.verify_token(token + "tampered") is None, "篡改签名必须失效"
    monkeypatch.setattr(auth, "TOKEN_TTL", -10)   # 签发即过期
    expired = auth.make_token("employee", "employee")
    assert auth.verify_token(expired) is None, "过期 token 必须失效"


def test_audit_records_queries(monkeypatch, tmp_path):
    monkeypatch.setattr(auth, "DB_PATH", tmp_path / "u.db")
    auth.seed()
    auth.audit("boss", "chat", "营收多少", ["机密.md"])
    entries = auth.audit_tail()
    assert entries and entries[0]["question"] == "营收多少" and "机密.md" in entries[0]["sources"]


def test_set_visibility_runtime_effect(acl_env):
    """运行时改密级：L4→L1 员工立即可见，改回 L4 立即消失（无需重灌）。"""
    acl_env.set_source_visibility("机密.md", "L1")
    r1 = acl_env.ask("营收多少", mode="full", allowed=allowed_for("employee"))
    assert any(c["source"] == "机密.md" for c in r1.citations), "降密后员工应能检索到"

    acl_env.set_source_visibility("机密.md", "L4")
    r2 = acl_env.ask("营收多少", mode="full", allowed=allowed_for("employee"))
    assert all(c["source"] != "机密.md" for c in r2.citations), "升密后员工应立即检索不到"


def test_set_visibility_missing_doc_raises(acl_env):
    with pytest.raises(ValueError):
        acl_env.set_source_visibility("不存在的文档.md", "L1")


def test_corrective_requery_forwards_allowed(acl_env, monkeypatch):
    """回归：CRAG 纠正重查扩大候选集时也必须沿用 ACL，否则机密文档会漏给低权限角色。"""
    from types import SimpleNamespace

    monkeypatch.setattr(settings, "enable_corrective", True)
    monkeypatch.setattr(settings, "rerank_provider", "stub", raising=False)
    calls = []

    def fake_retrieve(queries, mode="full", top_k=None, candidate_k=None, allowed=None):
        calls.append(allowed)
        return [SimpleNamespace(score=0.9, source="公开.md")], SimpleNamespace()

    monkeypatch.setattr(acl_env.retriever, "retrieve", fake_retrieve)
    oq = SimpleNamespace(original="营收多少", recall_queries=["营收多少", "营收"])
    hits = [SimpleNamespace(score=0.1, source="公开.md")]

    out, corrected = acl_env._maybe_corrective(oq, hits, "full", 5, allowed=["L1"])
    assert corrected, "低置信度时应触发一次纠正重查"
    assert calls == [["L1"]], "纠正重查必须带上 allowed，不能退回全量可见"
    assert out[0].score == 0.9
