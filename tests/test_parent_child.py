"""Parent-Child 分块：小块检索命中后合并为章节父块作为生成上下文。"""
import textwrap

from app.config import settings


def _ingest(rag_env, tmp_path):
    doc = tmp_path / "手册.md"
    doc.write_text(textwrap.dedent("""\
        # 手册
        ## 保修政策
        """ + "整机保修两年，电机保修五年，滤芯属于耗材不提供保修服务。" * 40 + """
        ## 售后电话
        客服热线 400-000-0000，服务时间每天九点到十八点。
        """), encoding="utf-8")
    return rag_env.ingest_file(doc)


def test_ingest_populates_parent_store(rag_env, tmp_path):
    _ingest(rag_env, tmp_path)
    hits, _ = rag_env.retriever.retrieve(["保修"], mode="hybrid", top_k=5)
    assert hits and all(h.parent_id for h in hits)
    info = rag_env.parent_store.get(hits[0].parent_id)
    assert info and info["section_path"] == "保修政策"
    assert len(info["text"]) > len(hits[0].text)  # 父块 = 章节全文，长于子块


def test_parent_mode_merges_same_section_hits(rag_env, tmp_path, monkeypatch):
    _ingest(rag_env, tmp_path)
    monkeypatch.setattr(settings, "enable_parent_child", True)
    result = rag_env.ask("保修政策是什么？", mode="full")
    # 同一章节的多个命中子块应合并为一个父块引用
    pids = [c.get("parent_id") for c in result.citations if c.get("parent_id")]
    assert len(pids) == len(set(pids)), "父块引用不应重复"
    assert result.citations, "答案应带引用"


def test_default_off_keeps_chunk_level(rag_env, tmp_path):
    _ingest(rag_env, tmp_path)
    result = rag_env.ask("保修政策是什么？", mode="full")
    assert all(c.get("parent_id") for c in result.citations)
    assert all(c["text"] for c in result.citations)
    # 关闭时：命中块保持子块粒度（不等于父块全文长度）
    result2 = rag_env.ask("售后电话多少？", mode="hybrid")
    assert all(len(c["text"]) < 1500 for c in result2.citations)


def test_delete_source_cleans_parents(rag_env, tmp_path):
    result = _ingest(rag_env, tmp_path)
    hits, _ = rag_env.retriever.retrieve(["保修"], mode="naive", top_k=3)
    pid = hits[0].parent_id
    rag_env.delete_source(result["source"])
    assert rag_env.parent_store.get(pid) is None
