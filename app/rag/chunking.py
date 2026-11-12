"""三级递归字符切分器（自实现，参考 LangChain RecursiveCharacterTextSplitter 思路）：

1. 按分隔符优先级递归下钻：段落 > 行 > 中文句 > 英文句 > 词 > 硬截断
2. 相邻 chunk 之间保留 overlap，避免语义被切断
3. 支持上下文块头：把「文档名 > 章节路径」拼到块前，提升检索命中率
"""
from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ". ", " ", ""]


@dataclass
class Chunk:
    text: str            # 原始正文（用于展示与引用）
    embedding_text: str  # 拼了上下文块头的文本（用于向量化/BM25）
    metadata: dict = field(default_factory=dict)


def _split_by_sep(text: str, sep: str) -> list[str]:
    """按分隔符切分并保留分隔符（挂在片段尾部），避免句子信息丢失。"""
    if sep == "":
        return [text[i:i + 1] for i in range(len(text))] if text else []
    parts = text.split(sep)
    out: list[str] = []
    for i, part in enumerate(parts):
        piece = part + sep if i < len(parts) - 1 else part
        if piece:
            out.append(piece)
    return [p for p in out if p.strip()]


def _merge_pieces(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """把小片段合并成接近 chunk_size 的块，相邻块保留 overlap 尾部。"""
    chunks: list[str] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = buf[-overlap:] if overlap > 0 else ""

    for piece in pieces:
        if len(buf) + len(piece) > chunk_size and buf:
            flush()
        buf += piece
        # 单片段自身超长：立即硬切（保底，正常流程中 recursive_split 已处理）
        while len(buf) > chunk_size * 2:
            chunks.append(buf[:chunk_size])
            buf = buf[chunk_size - overlap:]
    flush()
    return chunks


def recursive_split(
    text: str,
    chunk_size: int = 500,
    overlap: int = 80,
    separators: list[str] | None = None,
) -> list[str]:
    """递归字符切分：当前层切出的片段仍超长，就用下一级分隔符继续切；
    大小合格的相邻片段跨片合并成块，保证块内语义连贯、块间有重叠。"""
    seps = separators if separators is not None else DEFAULT_SEPARATORS
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    sep, rest = seps[0], seps[1:]
    pieces = _split_by_sep(text, sep)
    chunks: list[str] = []
    small: list[str] = []  # 积攒大小合格的片段，统一合并以应用 overlap
    for piece in pieces:
        if len(piece) <= chunk_size:
            small.append(piece)
        else:
            if small:
                chunks.extend(_merge_pieces(small, chunk_size, overlap))
                small = []
            chunks.extend(recursive_split(piece, chunk_size, overlap, rest))
    if small:
        chunks.extend(_merge_pieces(small, chunk_size, overlap))
    return [c for c in chunks if c.strip()]


def contextual_header(title: str, section_path: str) -> str:
    """生成上下文块头：「文档名 > 章节 > 小节」。"""
    parts = [p for p in (title, section_path) if p and p.strip()]
    return " > ".join(parts)


def chunk_document(
    text: str,
    title: str,
    section_path: str = "",
    page: int | None = None,
    source: str = "",
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[Chunk]:
    """把一份（章节级）文本切成带元数据与上下文块头的 Chunk 列表。"""
    header = contextual_header(title, section_path)
    pieces = recursive_split(text, chunk_size, overlap)
    chunks: list[Chunk] = []
    for idx, piece in enumerate(pieces):
        meta = {
            "source": source or title,
            "title": title,
            "section_path": section_path,
            "chunk_index": idx,
        }
        if page is not None:
            meta["page"] = page
        chunks.append(
            Chunk(
                text=piece,
                embedding_text=f"{header}\n{piece}" if header else piece,
                metadata=meta,
            )
        )
    return chunks
