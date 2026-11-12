"""文档解析器：把 md / txt / pdf / docx 统一解析为「章节片段」列表。

每个片段 = {text, title, section_path, page}，之后交给 chunk_document 切分。
markdown 与 docx 利用标题层级构建「文档名 > 章节」路径，pdf 以页为单位。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Section:
    text: str
    title: str
    section_path: str
    page: int | None = None


class UnsupportedFileTypeError(ValueError):
    pass


def _md_heading_level(line: str) -> int | None:
    m = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
    return len(m.group(1)) if m else None


def load_markdown(path: Path) -> list[Section]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    lines = raw.splitlines()
    # 文档主标题：第一个 H1，否则用文件名
    title = path.stem
    for line in lines:
        if _md_heading_level(line) == 1:
            title = re.sub(r"^#\s+", "", line).strip()
            break

    sections: list[Section] = []
    stack: dict[int, str] = {}  # 标题层级 -> 标题文本
    buf: list[str] = []
    current_path = ""

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            sections.append(Section(text=text, title=title, section_path=current_path))
        buf.clear()

    for line in lines:
        level = _md_heading_level(line)
        if level:
            flush()
            heading = re.sub(r"^#+\s+", "", line).strip()
            stack[level] = heading
            # 清掉更深层的旧标题
            for l in [l for l in stack if l > level]:
                del stack[l]
            current_path = " > ".join(stack[k] for k in sorted(stack) if k > 1)
        else:
            buf.append(line)
    flush()
    return sections


def load_txt(path: Path) -> list[Section]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [Section(text=text, title=path.stem, section_path="")]


def load_pdf(path: Path) -> list[Section]:
    import pymupdf  # PyMuPDF

    doc = pymupdf.open(str(path))
    meta_title = (doc.metadata or {}).get("title") or path.stem
    sections: list[Section] = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()  # type: ignore[attr-defined]
        if text:
            sections.append(
                Section(text=text, title=meta_title, section_path=f"第{i + 1}页", page=i + 1)
            )
    doc.close()
    return sections


def load_docx(path: Path) -> list[Section]:
    from docx import Document

    doc = Document(str(path))
    title = doc.core_properties.title or path.stem
    sections: list[Section] = []
    stack: dict[int, str] = {}
    buf: list[str] = []
    current_path = ""

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            sections.append(Section(text=text, title=title, section_path=current_path))
        buf.clear()

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "").lower()
        level = None
        if style.startswith("heading"):
            try:
                level = int(style.replace("heading", "").strip() or "1")
            except ValueError:
                level = None
        if level:
            flush()
            stack[level] = text
            for l in [l for l in stack if l > level]:
                del stack[l]
            current_path = " > ".join(stack[k] for k in sorted(stack) if k > 1)
        else:
            buf.append(text)
    flush()
    return sections


LOADERS = {
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_txt,
    ".pdf": load_pdf,
    ".docx": load_docx,
}


def load_file(path: str | Path) -> list[Section]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix not in LOADERS:
        supported = "、".join(sorted(LOADERS))
        raise UnsupportedFileTypeError(f"暂不支持的格式 {suffix}（支持：{supported}）")
    return LOADERS[suffix](p)
