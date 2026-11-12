"""视觉识图测试：PDF 抽图、识图描述入库（描述用 mock，不打真实 API）。"""
import io

import pytest

from app.config import settings
from app.rag import vision


def _make_image_pdf(tmp_path):
    """生成一份"文字 + 内嵌图表"的测试 PDF。"""
    import pymupdf
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (400, 300), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, 360, 260], fill=(200, 220, 255))
    d.text((60, 130), "Total: 5000 USD", fill="black")
    buf = io.BytesIO()
    img.save(buf, "PNG")

    pdf = tmp_path / "报表.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 60), "测试报表", fontsize=18)
    page.insert_image(pymupdf.Rect(72, 100, 472, 400), stream=buf.getvalue())
    doc.save(str(pdf))
    doc.close()
    return pdf


def test_extract_images_from_pdf(tmp_path):
    pdf = _make_image_pdf(tmp_path)
    imgs = vision.extract_images(pdf)
    assert len(imgs) == 1
    assert imgs[0]["page"] == 1
    assert len(imgs[0]["png_bytes"]) > 1000


def test_extract_md_returns_empty(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("# 纯文本", encoding="utf-8")
    assert vision.extract_images(f) == []


def test_ingest_creates_image_chunks(rag_env, tmp_path, monkeypatch):
    pdf = _make_image_pdf(tmp_path)
    monkeypatch.setattr(settings, "vision_enabled", True)
    monkeypatch.setattr(vision, "describe_image",
                        lambda png: "图表显示 Total 5000 USD 的金额汇总信息。")
    result = rag_env.ingest_file(pdf)
    assert result["images"] == 1 and result["chunks"] >= 2
    # 图片描述 chunk 可被检索到，且章节路径标记为"图片"
    hits, _ = rag_env.retriever.retrieve(["Total 5000 USD"], mode="hybrid", top_k=5)
    assert any(h.section_path.startswith("图片") for h in hits)
    assert all(h.visibility == "L1" for h in hits), "图片描述 chunk 继承文档密级"


def test_describe_none_skips_image(rag_env, tmp_path, monkeypatch):
    pdf = _make_image_pdf(tmp_path)
    monkeypatch.setattr(settings, "vision_enabled", True)
    monkeypatch.setattr(vision, "describe_image", lambda png: None)   # 模型称"无有效内容"
    result = rag_env.ingest_file(pdf)
    assert result["images"] == 0
    hits, _ = rag_env.retriever.retrieve(["Total 5000 USD"], mode="hybrid", top_k=5)
    assert all(not h.section_path.startswith("图片") for h in hits)


def test_vision_disabled_skips(rag_env, tmp_path, monkeypatch):
    pdf = _make_image_pdf(tmp_path)
    monkeypatch.setattr(settings, "vision_enabled", False)
    monkeypatch.setattr(vision, "describe_image",
                        lambda png: pytest.fail("关闭识图时不应调用视觉模型"))
    result = rag_env.ingest_file(pdf)
    assert result["images"] == 0


def test_image_file_ingest_via_describe(rag_env, tmp_path, monkeypatch):
    """A 功能：直接上传图片文件 → 识图描述作为唯一 chunk 入库。"""
    import io

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (320, 200), "white")
    d = ImageDraw.Draw(img)
    d.text((20, 80), "Project Atlas: revenue 88M", fill="black")
    buf = io.BytesIO()
    img.save(buf, "PNG")

    f = tmp_path / "atlas.png"
    f.write_bytes(buf.getvalue())
    monkeypatch.setattr(vision, "describe_image",
                        lambda png: "图片内容：Project Atlas 报告，revenue 88M。")
    result = rag_env.ingest_file(f)
    assert result["chunks"] == 1 and result["images"] == 1
    hits, _ = rag_env.retriever.retrieve(["Atlas revenue 88M"], mode="hybrid", top_k=3)
    assert any(h.source == "atlas.png" for h in hits)


def test_invalid_image_file_rejected(rag_env, tmp_path):
    """非法图片字节：to_png_bytes 应抛异常（API 层转 400/500），不能静默入库。"""
    f = tmp_path / "fake.png"
    f.write_bytes(b"this is not an image")
    with pytest.raises(Exception):
        rag_env.ingest_file(f)
