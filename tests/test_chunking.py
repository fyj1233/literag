"""切分器单测：边界、重叠、上下文块头。"""
from app.rag.chunking import chunk_document, recursive_split


def test_short_text_single_chunk():
    assert recursive_split("你好世界") == ["你好世界"]


def test_empty_text():
    assert recursive_split("") == []
    assert recursive_split("   \n  ") == []


def test_long_text_respects_size():
    text = "这是一个测试句子。" * 200  # 约 1800 字
    chunks = recursive_split(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 220 for c in chunks)  # 允许少量边界余量


def test_adjacent_chunks_have_overlap():
    text = "".join(f"第{i}句话讲的是测试内容。" for i in range(80))
    chunks = recursive_split(text, chunk_size=120, overlap=30)
    # 相邻块首尾应存在内容衔接（overlap 生效）
    assert chunks[1][:30] in chunks[0][-60:] or chunks[0][-30:] in chunks[1][:60]


def test_paragraph_boundary_preferred():
    text = "段落一的内容。\n\n段落二的内容。\n\n" + "长" * 400
    chunks = recursive_split(text, chunk_size=100, overlap=10)
    # 短段落不应被拆散进"长"字块中
    assert any("段落一的内容" in c for c in chunks)
    assert any("段落二的内容" in c for c in chunks)


def test_hard_split_for_oversized_piece():
    text = "A" * 1000  # 无任何分隔符
    chunks = recursive_split(text, chunk_size=200, overlap=50)
    assert len(chunks) >= 4
    assert all(len(c) <= 200 for c in chunks)


def test_chunk_document_metadata():
    chunks = chunk_document(
        text="内容一。" * 100, title="测试文档", section_path="第一章 > 背景",
        source="test.md", chunk_size=100, overlap=10,
    )
    assert len(chunks) > 1
    first = chunks[0]
    assert first.metadata["source"] == "test.md"
    assert first.metadata["title"] == "测试文档"
    # embedding_text 应带上下文块头
    assert first.embedding_text.startswith("测试文档 > 第一章 > 背景")
    assert first.text in first.embedding_text
    # chunk_index 全局递增
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
