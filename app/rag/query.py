"""查询优化：LLM 查询改写 + 子查询分解（多查询）+ HyDE（可选）。

作用：用户的原始提问往往口语化、有省略或一问多事，直接拿去召回效果差。
这里用 LLM 把问题改写成更适合检索的陈述句，并拆出 2~3 个子问题分别召回，
再由 RRF 融合，显著提升多跳/复合问题的命中率。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.config import settings
from app.providers.llm import LLMProvider

REWRITE_PROMPT = """你是检索查询优化器。请针对下面的用户问题：
1. 改写为一个更适合在知识库中检索的完整陈述句（补全省略、消除口语）；
2. 如果问题包含多个子话题或需要多步推理，拆分出最多 2 个子问题；否则子问题列表为空。

只输出 JSON，格式：
{{"rewritten": "改写后的问题", "sub_queries": ["子问题1", "子问题2"]}}

用户问题：{question}"""

HISTORY_PROMPT = """【对话历史】
{history}

注意：若用户问题包含指代（如"它""第二年""上面说的"），请结合对话历史将问题消解为不依赖上下文的独立问题。

"""


def _history_text(history: list[dict] | None) -> str:
    if not history:
        return ""
    lines = [
        f"{'用户' if m.get('role') == 'user' else '助手'}: {str(m.get('content', ''))[:200]}"
        for m in history[-4:]  # 最多带两轮历史，控制上下文长度
    ]
    return "\n".join(lines)


@dataclass
class OptimizedQuery:
    original: str
    rewritten: str = ""
    sub_queries: list[str] = field(default_factory=list)
    hyde_doc: str = ""

    @property
    def recall_queries(self) -> list[str]:
        """实际参与召回的查询列表：改写主查询优先，附子查询与 HyDE 假想文档。"""
        qs = [self.rewritten or self.original]
        qs.extend(q for q in self.sub_queries if q and q not in qs)
        if self.hyde_doc:
            qs.append(self.hyde_doc)
        return qs


def optimize_query(
    question: str, llm: LLMProvider, enable: bool = True, history: list[dict] | None = None
) -> OptimizedQuery:
    """改写 + 分解（多轮时结合历史做指代消解）。LLM 失败时优雅降级为原始问题。"""
    oq = OptimizedQuery(original=question)
    if not enable or not settings.enable_query_rewrite:
        return oq
    try:
        prompt = REWRITE_PROMPT.format(question=question)
        hist = _history_text(history)
        if hist:
            prompt = HISTORY_PROMPT.format(history=hist) + prompt
        data = llm.chat_json([{"role": "user", "content": prompt}])
        oq.rewritten = str(data.get("rewritten", "")).strip() or question
        subs = data.get("sub_queries", [])
        if isinstance(subs, list) and settings.enable_multi_query:
            oq.sub_queries = [str(s).strip() for s in subs if str(s).strip()][:2]
    except Exception:  # noqa: BLE001 —— 改写失败不影响主流程
        oq.rewritten = question
    return oq


def hyde_document(question: str, llm: LLMProvider) -> str:
    """HyDE：先生成一段假想答案，用它去检索（假想文档与真实文档的向量更接近）。"""
    if not settings.enable_hyde:
        return ""
    try:
        return llm.chat(
            [{"role": "user", "content": f"请用一段话直接回答以下问题（不需要引用，200字内）：\n{question}"}]
        ).strip()
    except Exception:  # noqa: BLE001
        return ""
