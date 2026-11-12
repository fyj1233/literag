"""视觉识图模块：抽取文档内嵌图片 → 识图模型生成中文描述 → 供摄入管道入库。

设计：
1. extract_images(path)：PDF 用 PyMuPDF 逐页抽取内嵌图；DOCX 遍历图片关系部件；
   md/txt 无图片。小尺寸图（图标/装饰）自动跳过。
2. describe_image(png_bytes)：调用"识图模型"（默认复用主 LLM 端点，要求其支持多模态；
   可在设置面板/`.env` 单独配置 VISION_*）生成中文描述。
3. 全程优雅降级：识图失败/未启用/超限都不阻断摄入主链路。
"""
from __future__ import annotations

import base64
from pathlib import Path

from app.config import settings

MIN_PIXELS = 100          # 宽或高低于该值的图视为图标/装饰，跳过
DESCRIBE_PROMPT = (
    "请用中文客观、详细地描述这张图片中的全部信息，包括：图表类型与数据、坐标轴、"
    "图内文字、流程或结构关系。描述将用于知识库检索，请把关键数字和结论写清楚。"
    "如果图片不包含有效信息（纯装饰/Logo），只回答：无有效内容"
)


def to_png_bytes(raw: bytes) -> bytes:
    """任意图片字节 → 统一转成 PNG（同时验证这是合法图片，非法抛异常）。"""
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(raw))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "PNG")
    return buf.getvalue()


def extract_images(file_path: str | Path) -> list[dict]:
    """抽取文档内嵌图片，返回 [{png_bytes, page}]（page 为 None 表示未知页码）。"""
    path = Path(file_path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            return _extract_pdf(path)
        if suffix == ".docx":
            return _extract_docx(path)
    except Exception:  # noqa: BLE001 —— 抽图失败不阻断摄入
        return []
    return []


def _extract_pdf(path: Path) -> list[dict]:
    import pymupdf

    out: list[dict] = []
    with pymupdf.open(str(path)) as doc:
        for page_no, page in enumerate(doc, start=1):
            for img in page.get_images(full=True):
                try:
                    pix = pymupdf.Pixmap(doc, img[0])
                    if pix.width < MIN_PIXELS or pix.height < MIN_PIXELS:
                        continue
                    if pix.n - pix.alpha > 3:   # CMYK 等转 RGB
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                    out.append({"png_bytes": pix.tobytes("png"), "page": page_no})
                except Exception:  # noqa: BLE001 —— 单张图损坏跳过
                    continue
    return out


def _extract_docx(path: Path) -> list[dict]:
    from docx import Document

    doc = Document(str(path))
    out: list[dict] = []
    for rel in doc.part.rels.values():
        if "image" not in rel.reltype:
            continue
        try:
            blob = rel.target_part.blob
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(blob))
            if img.width < MIN_PIXELS or img.height < MIN_PIXELS:
                continue
            buf = io.BytesIO()
            img.convert("RGB").save(buf, "PNG")
            out.append({"png_bytes": buf.getvalue(), "page": None})
        except Exception:  # noqa: BLE001 —— 单张图损坏跳过
            continue
    return out


def describe_image(png_bytes: bytes) -> str | None:
    """调用识图模型生成中文描述；失败重试 2 次，仍失败返回 None（调用方跳过该图）。"""
    import logging
    import time as _time

    from openai import OpenAI

    logger = logging.getLogger("literag.vision")

    base_url = settings.vision_base_url or settings.llm_base_url
    api_key = settings.vision_api_key or settings.llm_api_key
    model = settings.vision_model or settings.llm_model
    if not api_key:
        return None

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=90)
    b64 = base64.b64encode(png_bytes).decode()

    # 推理型识图模型会把 max_tokens 预算花在内部思考上（content 为空 + finish_reason=length），
    # 因此预算不足时自动翻倍重试（2000→4000→8000）
    budget = 2000
    for attempt in (1, 2, 3):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=budget,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": DESCRIBE_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]}],
            )
            choice = resp.choices[0]
            text = (choice.message.content or "").strip()
            if choice.finish_reason == "length" and not text:
                logger.warning("识图输出为空（推理占满 %d 预算），加大预算重试", budget)
                budget *= 2
                continue
        except Exception as e:  # noqa: BLE001 —— 网络类失败重试一次，仍失败则降级跳过
            logger.warning("识图失败(第%d次): %s", attempt, e)
            _time.sleep(1)
            continue
        if not text or "无有效内容" in text:
            return None
        return text
    logger.warning("识图重试耗尽，跳过该图片")
    return None
