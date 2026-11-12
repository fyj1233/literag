"""生成层：组装 Prompt、调用 LLM 流式生成、答案与引用编号对齐。"""
from __future__ import annotations

from collections.abc import Iterator

from app.providers.llm import CONTEXT_MARKER_BEGIN, CONTEXT_MARKER_END, LLMProvider
from app.rag.retrieval import extract_citations
from app.rag.storage import Hit

SYSTEM_PROMPT = """你是企业知识库问答助手，请严格遵守：
1. 只依据【参考资料】中的内容回答，不要编造资料之外的 信息；
2. 回答中引用某条资料时，在句末标注其编号，如 [1]、[2]，编号必须与资料编号一一对应；
3. 若资料不足以回答问题，直接说明"知识库中没有找到相关内容"，不要猜测；
4. 用简体中文，简洁、分点作答。"""


def build_context(hits: list[Hit]) -> str:
    """把召回的 chunk 组装成带编号的资料块：[1] 正文 …"""
    lines = [CONTEXT_MARKER_BEGIN]
    for i, h in enumerate(hits, start=1):
        location = f"（来源: {h.source}"
        if h.section_path:
            location += f" > {h.section_path}"
        if h.page is not None:
            location += f" 第{h.page}页"
        location += "）"
        lines.append(f"[{i}] {h.text.strip()}\n{location}")
    lines.append(CONTEXT_MARKER_END)
    return "\n".join(lines)


def build_messages(question: str, hits: list[Hit], history: list[dict] | None = None,
                   image_data_url: str | None = None) -> list[dict]:
    """组装对话消息。image_data_url 非空时，用户消息升级为多模态（文本+图片）。"""
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-4:])  # 最多带两轮历史，控制上下文长度
    user_content = f"{build_context(hits)}\n\n【用户问题】{question}"
    if image_data_url:
        user_content = [
            {"type": "text", "text": user_content},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]
    messages.append({"role": "user", "content": user_content})
    return messages


def align_citations(answer: str, hits: list[Hit]) -> list[dict]:
    """提取答案中实际使用的引用编号，映射回 chunk；越界编号忽略。"""
    used = extract_citations(answer)
    out: list[dict] = []
    for n in used:
        if 1 <= n <= len(hits):
            item = hits[n - 1].to_dict()
            item["ref"] = n
            out.append(item)
    return out


def stream_answer(llm: LLMProvider, messages: list[dict]) -> Iterator[str]:
    yield from llm.chat_stream(messages)
