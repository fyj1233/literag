"""检索评估指标（不需要 LLM，可完全离线复现）：

- Hit@k  : 期望来源文档是否出现在 top-k 命中的来源集合中（期望来源支持多个）
- MRR@k  : 第一条命中期望来源的结果排名倒数（0 = 未命中）
"""


def _as_list(expected) -> list[str]:
    return [expected] if isinstance(expected, str) else list(expected)


def hit_at_k(hits: list[dict], expected_source, k: int = 5) -> float:
    sources = {h["source"] for h in hits[:k]}
    return 1.0 if any(s in sources for s in _as_list(expected_source)) else 0.0


def mrr_at_k(hits: list[dict], expected_source, k: int = 10) -> float:
    wanted = set(_as_list(expected_source))
    for rank, h in enumerate(hits[:k], start=1):
        if h["source"] in wanted:
            return 1.0 / rank
    return 0.0


def keyword_coverage(hits: list[dict], keywords: list[str], k: int = 5) -> float:
    """辅助指标：top-k 命中文本对关键词的覆盖比例。"""
    if not keywords:
        return 0.0
    text = " ".join(h["text"] for h in hits[:k])
    return sum(1 for kw in keywords if kw in text) / len(keywords)
