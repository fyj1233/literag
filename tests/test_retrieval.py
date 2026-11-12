"""RRF 融合与引用对齐单测。"""
from app.rag.retrieval import extract_citations, rrf_fuse


def test_rrf_basic_ordering():
    # 同时出现在两路前列的元素应排最前
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    ids = [i for i, _ in fused]
    assert ids[0] in ("a", "b")
    assert set(ids) == {"a", "b", "c", "d"}


def test_rrf_double_hit_beats_single():
    fused = rrf_fuse([["x", "p", "q"], ["y", "p", "q"]], k=60)
    scores = dict(fused)
    # p 在两路都是第 2 名，q 在两路都是第 3 名 → p > q > x/y
    assert scores["p"] > scores["q"]
    assert scores["q"] > scores["x"] and scores["q"] > scores["y"]


def test_rrf_rank_position_matters():
    fused = rrf_fuse([["a", "b"], ["a", "b"]], k=60)
    scores = dict(fused)
    assert scores["a"] > scores["b"]


def test_extract_citations():
    assert extract_citations("根据[1]和[3]的说法[2]") == [1, 2, 3]
    assert extract_citations("重复[1]再次[1]") == [1]
    assert extract_citations("没有引用") == []
    assert extract_citations("越界[99]与[5]") == [5, 99]
