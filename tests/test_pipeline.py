"""端到端流水线测试：摄入 → 检索 → mock 生成 → 引用对齐（全离线）。"""
import textwrap


def _make_doc(tmp_path, name="测试知识.md"):
    path = tmp_path / name
    path.write_text(
        textwrap.dedent("""\
        # 测试产品手册
        ## 保修政策
        该产品整机保修两年，电机保修五年。滤芯属于耗材不提供保修。
        ## 售后电话
        客服热线是 400-000-0000，工作时间为每天九点到十八点。
        """),
        encoding="utf-8",
    )
    return path


def test_ingest_and_stats(rag_env, tmp_path):
    result = rag_env.ingest_file(_make_doc(tmp_path))
    assert result["chunks"] >= 2
    stats = rag_env.stats()
    assert stats["chunks"] == result["chunks"]
    assert "测试知识.md" in stats["sources"]


def test_reingest_is_idempotent(rag_env, tmp_path):
    doc = _make_doc(tmp_path)
    rag_env.ingest_file(doc)
    first = rag_env.vector_store.count()
    rag_env.ingest_file(doc)  # 重传应删除旧索引后再写入
    assert rag_env.vector_store.count() == first


def test_ask_returns_answer_with_citations(rag_env, tmp_path):
    rag_env.ingest_file(_make_doc(tmp_path))
    result = rag_env.ask("保修期是多久？", mode="full")
    assert "保修" in result.answer
    assert result.citations, "答案应带引用"
    assert result.citations[0]["source"] == "测试知识.md"
    assert result.trace.vector_hits > 0


def test_ask_unanswerable_question_falls_back(rag_env, tmp_path):
    rag_env.ingest_file(_make_doc(tmp_path))
    result = rag_env.ask("量子力学的波函数坍缩如何理解？", mode="naive")
    # mock 模式下未命中关键词应给出兜底话术，而不是编造
    assert "没有检索到" in result.answer or result.citations


def test_delete_source(rag_env, tmp_path):
    rag_env.ingest_file(_make_doc(tmp_path))
    deleted = rag_env.delete_source("测试知识.md")
    assert deleted > 0
    assert rag_env.vector_store.count() == 0
    assert rag_env.bm25_store.search("保修", 5) == []
