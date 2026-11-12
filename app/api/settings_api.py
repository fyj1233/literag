"""boss 专属设置面板 API：LLM 端点热更新 + 测速、功能开关、检索参数、审计日志。

安全边界：
- 全部接口 require_role("boss")
- api_key 永远以掩码返回（sk-****后四位），完整 key 只在保存时提交、写入 .env（gitignored）
- Embedding/Rerank 配置**只读**——切换即换向量维度，需要全库重灌，不允许"点一下"发生
"""
from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app import auth
from app.api.deps import require_role
from app.config import persist_env, settings
from app.rag.pipeline import get_pipeline

router = APIRouter(prefix="/api/settings", dependencies=[Depends(require_role("boss"))])

FEATURE_NAMES = [
    "enable_hybrid", "enable_rerank", "enable_query_rewrite", "enable_multi_query",
    "enable_hyde", "enable_corrective", "enable_parent_child",
]
INT_PARAMS = {
    "vector_top_k": (1, 50), "bm25_top_k": (1, 50), "rrf_k": (1, 200),
    "candidate_k": (5, 100), "final_top_k": (1, 20),
    "chunk_size": (100, 2000), "chunk_overlap": (0, 400),
}
REBUILD_CHUNK_PARAMS = {"chunk_size", "chunk_overlap"}   # 改动需重灌索引


def mask_key(key: str) -> str:
    return key[:4] + "****" + key[-4:] if len(key) >= 10 else "****"


def _llm_view() -> dict:
    key = settings.llm_api_key
    return {
        "provider": settings.llm_provider,
        "base_url": settings.llm_base_url,
        "api_key_masked": mask_key(key) if key else "",
        "has_key": bool(key),
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
    }


class LLMSettings(BaseModel):
    base_url: str = Field(min_length=1)
    api_key: str = ""        # 留空 = 保持现有 key 不变
    model: str = Field(min_length=1)
    temperature: float = Field(ge=0, le=2)
    max_tokens: int = Field(ge=64, le=8192)


class LLMTestRequest(BaseModel):
    """可选字段：不传则测当前已保存配置；传了则只测候选配置（不保存）。"""
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None


class VisionTestRequest(BaseModel):
    """识图连通性测试：上传一张测试图，验证端点/密钥/模型是否真的能看图。"""
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None


@router.get("")
def get_settings_view() -> dict:
    return {
        "llm": _llm_view(),
        "vision": _vision_view(),
        "features": {name: getattr(settings, name) for name in FEATURE_NAMES},
        "retrieval": {name: getattr(settings, name) for name in INT_PARAMS},
        "read_only": {
            "embedding": f"{settings.embedding_provider}",
            "reranker": settings.rerank_provider,
            "note": "Embedding/Rerank 切换会改变向量维度或需全库重灌，请编辑 .env 后重灌索引",
        },
    }


def _vision_view() -> dict:
    """识图模型视图：三项留空表示复用主模型；掩码显示生效中的密钥。"""
    uses_main = not (settings.vision_base_url or settings.vision_api_key or settings.vision_model)
    effective_key = settings.vision_api_key or settings.llm_api_key
    return {
        "enabled": settings.vision_enabled,
        "uses_main": uses_main,
        "base_url": settings.vision_base_url,
        "api_key_masked": mask_key(effective_key) if effective_key else "",
        "model": settings.vision_model or f"（复用主模型 {settings.llm_model}）",
        "max_images": settings.vision_max_images,
    }


class VisionSettings(BaseModel):
    enabled: bool
    base_url: str = ""       # 留空 = 复用主模型端点
    api_key: str | None = None   # None = 保持不变；空串 = 切换为复用主模型密钥
    model: str = ""          # 留空 = 复用主模型
    max_images: int = Field(default=5, ge=0, le=50)


@router.put("/vision")
def put_vision(body: VisionSettings) -> dict:
    """保存识图模型配置：三项留空 = 复用主 LLM 端点/密钥/模型（要求主模型支持识图）。"""
    settings.vision_enabled = body.enabled
    settings.vision_base_url = body.base_url.strip()
    settings.vision_model = body.model.strip()
    settings.vision_max_images = body.max_images
    if body.api_key is not None:
        settings.vision_api_key = body.api_key.strip()
    updates = {
        "VISION_ENABLED": "1" if body.enabled else "0",
        "VISION_BASE_URL": body.base_url.strip(),
        "VISION_MODEL": body.model.strip(),
        "VISION_MAX_IMAGES": str(body.max_images),
    }
    if body.api_key is not None:
        updates["VISION_API_KEY"] = body.api_key.strip()
    persist_env(updates)
    return _vision_view()


@router.put("/llm")
def put_llm(body: LLMSettings) -> dict:
    """保存 LLM 端点配置：热更新内存对象 + 原子写回 .env + 重建 Provider。"""
    updates = {
        "LLM_PROVIDER": "openai",
        "LLM_BASE_URL": body.base_url.strip(),
        "LLM_MODEL": body.model.strip(),
        "LLM_TEMPERATURE": str(body.temperature),
        "LLM_MAX_TOKENS": str(body.max_tokens),
    }
    if body.api_key.strip():
        updates["LLM_API_KEY"] = body.api_key.strip()
        settings.llm_api_key = body.api_key.strip()
    settings.llm_provider = "openai"
    settings.llm_base_url = body.base_url.strip()
    settings.llm_model = body.model.strip()
    settings.llm_temperature = body.temperature
    settings.llm_max_tokens = body.max_tokens

    persist_env(updates)
    _rebuild_llm()   # 立即生效：下一次提问就走新端点
    return _llm_view()


@router.post("/llm/test")
def test_llm(body: LLMTestRequest | None = None) -> dict:
    """流式测速：优先测请求里带来的候选配置（保存前先试），否则测当前配置。"""
    base_url = (body.base_url if body and body.base_url else settings.llm_base_url)
    api_key = (body.api_key if body and body.api_key else settings.llm_api_key)
    model = (body.model if body and body.model else settings.llm_model)
    if not api_key:
        raise HTTPException(400, "未配置 API Key")

    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    t0 = time.perf_counter()
    first = None
    deltas = 0
    chars = 0
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "请用一句话回答：你是什么模型？"}],
            stream=True,
            max_tokens=120,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                now = time.perf_counter()
                if first is None:
                    first = now - t0
                deltas += 1
                chars += len(chunk.choices[0].delta.content)
    except Exception as e:  # noqa: BLE001 —— 测速失败的原始错误要返回给前端
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    total = time.perf_counter() - t0
    return {"ok": True, "ttft_s": round(first or total, 2), "deltas": deltas,
            "chars": chars, "total_s": round(total, 2), "model": model}


@router.post("/vision/test")
def test_vision(body: VisionTestRequest | None = None) -> dict:
    """识图连通性测试：现场生成一张含文字的测试图，验证识图模型真的能看图并读出图内文字。"""
    base_url = (body.base_url if body and body.base_url else
                settings.vision_base_url or settings.llm_base_url)
    api_key = (body.api_key if body and body.api_key else
               settings.vision_api_key or settings.llm_api_key)
    model = (body.model if body and body.model else
             settings.vision_model or settings.llm_model)
    if not api_key:
        raise HTTPException(400, "未配置 API Key")

    import io

    from openai import OpenAI

    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (420, 200), "white")
        d = ImageDraw.Draw(img)
        d.ellipse([30, 40, 130, 140], fill=(220, 40, 40))
        d.rectangle([190, 60, 320, 160], fill=(40, 80, 220))
        d.text((30, 165), "SECRET CODE: 7391", fill="black")
        buf = io.BytesIO()
        img.save(buf, "PNG")
        png_bytes = buf.getvalue()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"生成测试图失败（需要 Pillow）：{e}") from e

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=90)
    b64 = base64.b64encode(png_bytes).decode()
    t0 = time.perf_counter()
    budget = 2000
    try:
        for _ in range(3):
            resp = client.chat.completions.create(
                model=model,
                max_tokens=budget,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": "图片里有什么形状、什么颜色、写了什么文字？"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]}],
            )
            choice = resp.choices[0]
            answer = (choice.message.content or "").strip()
            if choice.finish_reason == "length" and not answer:
                budget *= 2   # 推理占满预算 → 加大预算重试
                continue
            break
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    elapsed = round(time.perf_counter() - t0, 2)
    # 判定：能读出图内文字 "7391" 才算真正识图成功（防"只回话不看图"）
    ok = "7391" in answer
    return {"ok": ok, "elapsed_s": elapsed, "model": model, "answer": answer[:200],
            "note": "已正确读出图内文字 7391" if ok else "未检测到图内文字 7391，请人工核对回答"}


@router.put("/features")
def put_features(body: dict[str, bool]) -> dict:
    updates = {}
    for name, value in body.items():
        if name not in FEATURE_NAMES:
            raise HTTPException(400, f"未知开关: {name}")
        setattr(settings, name, bool(value))
        updates[name.upper()] = "1" if value else "0"
    persist_env(updates)
    return {name: getattr(settings, name) for name in FEATURE_NAMES}


@router.put("/retrieval")
def put_retrieval(body: dict[str, int]) -> dict:
    updates = {}
    needs_reingest = False
    for name, value in body.items():
        if name not in INT_PARAMS:
            raise HTTPException(400, f"未知参数: {name}")
        lo, hi = INT_PARAMS[name]
        if not (lo <= value <= hi):
            raise HTTPException(400, f"{name} 应在 {lo}~{hi} 之间")
        setattr(settings, name, value)
        updates[name.upper()] = str(value)
        needs_reingest |= name in REBUILD_CHUNK_PARAMS
    persist_env(updates)
    return {"saved": True, "needs_reingest": needs_reingest,
            "note": "chunk_size/chunk_overlap 改动需重灌索引后生效" if needs_reingest else ""}


@router.get("/audit")
def get_audit() -> dict:
    return {"entries": auth.audit_tail()}


def _rebuild_llm() -> None:
    """LLM 配置变化后重建 Provider（轻对象，毫秒级）。"""
    from app.providers.llm import get_llm_provider

    pipe = get_pipeline()
    pipe.llm = get_llm_provider()
