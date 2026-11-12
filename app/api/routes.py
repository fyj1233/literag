"""HTTP API：认证 / chat(SSE 流式) / search / documents / audit / stats / health。

鉴权模型：登录换 JWT（HS256）→ 请求带 Authorization: Bearer → 依赖注入解析出
用户角色 → 角色映射为可见密级 → 检索层按密级过滤（无权限内容不进入任何下游环节）。
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import auth
from app.api.deps import current_user, require_role
from app.config import ROLE_LEVELS, settings
from app.rag.pipeline import IMAGE_SUFFIX, get_pipeline

router = APIRouter(prefix="/api")

ALLOWED_SUFFIX = {".md", ".markdown", ".txt", ".pdf", ".docx"} | IMAGE_SUFFIX
VALID_VISIBILITY = {"L1", "L2", "L3", "L4"}


# ---------- 认证端点（开放） ----------
class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=1, max_length=100)


class RegisterRequest(LoginRequest):
    role: str = "employee"   # boss 登录后可创建账号；角色限定三选一


@router.post("/auth/login")
def api_login(req: LoginRequest) -> dict:
    result = auth.login(req.username, req.password)
    if not result:
        raise HTTPException(401, "用户名或密码错误")
    return result


@router.post("/auth/register")
def api_register(req: RegisterRequest, user: dict = Depends(require_role("boss"))) -> dict:
    """创建账号：仅 boss 可操作（首次启动已内置 boss 账号做引导）。"""
    if req.role not in ROLE_LEVELS:
        raise HTTPException(400, f"role 必须是 {'/'.join(ROLE_LEVELS)}")
    if auth.get_user(req.username):
        raise HTTPException(409, "用户名已存在")
    auth.create_user(req.username, req.password, req.role)
    return {"created": req.username, "role": req.role}


@router.get("/auth/me")
def api_me(user: dict = Depends(current_user)) -> dict:
    return user


# ---------- 请求模型 ----------
class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: str = "full"          # naive / hybrid / rerank / full
    top_k: int | None = None    # 覆盖默认 FINAL_TOP_K
    history: list[dict] = Field(default_factory=list)
    image: str | None = Field(default=None, max_length=6_000_000,
                              description="随问题附图（data:image/png;base64,...），需识图模型")


class SearchRequest(ChatRequest):
    pass


class VisibilityRequest(BaseModel):
    source: str = Field(min_length=1)
    visibility: str   # L1 / L2 / L3 / L4


def _validate_mode(mode: str) -> str:
    if mode not in ("naive", "hybrid", "rerank", "full"):
        raise HTTPException(400, f"mode 必须是 naive/hybrid/rerank/full，收到 {mode!r}")
    return mode


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "app": settings.app_name}


@router.get("/stats")
def stats(user: dict = Depends(current_user)) -> dict:
    return get_pipeline().stats()


@router.post("/search")
def search(req: SearchRequest, user: dict = Depends(current_user)) -> dict:
    """纯检索调试接口：不走 LLM，直接看召回与融合、精排结果（同样受 ACL 限制）。"""
    _validate_mode(req.mode)
    pipe = get_pipeline()
    from app.rag.query import optimize_query

    oq = optimize_query(req.question, pipe.llm, enable=req.mode == "full")
    hits, trace = pipe.retriever.retrieve(
        oq.recall_queries, mode=req.mode, top_k=req.top_k, allowed=user["allowed"]
    )
    auth.audit(user["username"], "search", req.question, [h.source for h in hits])
    return {
        "question": req.question,
        "rewritten": oq.rewritten if oq.rewritten != req.question else "",
        "sub_queries": oq.sub_queries,
        "trace": {
            "queries": trace.queries,
            "vector_hits": trace.vector_hits,
            "bm25_hits": trace.bm25_hits,
            "fused_candidates": trace.fused_candidates,
            "timings": trace.timings,
        },
        "hits": [h.to_dict() for h in hits],
    }


@router.post("/chat")
def chat(req: ChatRequest, user: dict = Depends(current_user)) -> StreamingResponse:
    """SSE 流式问答：meta（检索轨迹）→ delta*（增量文本）→ done（答案+引用）。"""
    _validate_mode(req.mode)
    pipe = get_pipeline()
    if pipe.vector_store.count() == 0:
        raise HTTPException(409, "知识库为空，请先上传文档或运行 scripts/ingest.py")

    def event_stream():
        sources_seen: list[str] = []
        question_for_audit = req.question + ("（附图）" if req.image else "")
        for event in pipe.ask_stream(req.question, mode=req.mode, history=req.history,
                                     top_k=req.top_k, allowed=user["allowed"],
                                     image_data_url=req.image):
            if event.get("type") == "meta":
                sources_seen = list({h["source"] for h in event.get("hits", [])})
                auth.audit(user["username"], "chat", question_for_audit, sources_seen)
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/documents")
def list_documents(user: dict = Depends(current_user)) -> dict:
    """文档列表按当前用户可见密级过滤（含每份文档的密级标签）。"""
    allowed = user["allowed"]
    pipe = get_pipeline()
    counts = pipe.vector_store.sources(allowed=allowed)
    # 附上每份文档的密级标签（取该文档任一 chunk 的 visibility）
    got = pipe.vector_store.collection.get(include=["metadatas"])
    vis_map: dict[str, str] = {}
    for meta in got["metadatas"]:
        src = (meta or {}).get("source", "unknown")
        vis_map.setdefault(src, (meta or {}).get("visibility", "L1"))
    return {
        "documents": [
            {"source": name, "chunks": n, "visibility": vis_map.get(name, "L1")}
            for name, n in counts.items()
        ]
    }


@router.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    visibility: str = Form("L1"),
    user: dict = Depends(require_role("boss", "employee")),
) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIX:
        raise HTTPException(400, f"不支持的格式 {suffix}，支持：md / txt / pdf / docx")
    if visibility not in VALID_VISIBILITY:
        raise HTTPException(400, f"密级必须是 {'/'.join(sorted(VALID_VISIBILITY))}")

    save_path = settings.upload_dir / f"{uuid.uuid4().hex[:8]}_{file.filename}"
    save_path.write_bytes(await file.read())
    try:
        result = get_pipeline().ingest_file(save_path, source_name=file.filename,
                                            visibility=visibility)
    except Exception as e:  # noqa: BLE001 —— 解析失败要把脏数据路径告诉前端
        raise HTTPException(500, f"解析失败：{e}") from e
    auth.audit(user["username"], "upload", file.filename or "", [visibility])
    return result


@router.delete("/documents")
def delete_document(source: str,
                    user: dict = Depends(require_role("boss", "employee"))) -> dict:
    deleted = get_pipeline().delete_source(source)
    if not deleted:
        raise HTTPException(404, f"未找到文档: {source}")
    auth.audit(user["username"], "delete", source, [])
    return {"deleted": deleted, "source": source}


@router.put("/documents/visibility")
def set_document_visibility(req: VisibilityRequest,
                            user: dict = Depends(require_role("boss"))) -> dict:
    """运行时调整已入库文档的密级（仅 boss）；立即生效，无需重灌。"""
    if req.visibility not in VALID_VISIBILITY:
        raise HTTPException(400, f"密级必须是 {'/'.join(sorted(VALID_VISIBILITY))}")
    try:
        chunks = get_pipeline().set_source_visibility(req.source, req.visibility)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
    auth.audit(user["username"], "visibility", req.source, [req.visibility])
    return {"source": req.source, "visibility": req.visibility, "chunks": chunks}


@router.get("/audit")
def get_audit(user: dict = Depends(require_role("boss"))) -> dict:
    """审计日志：仅 boss 可查看最近的问答与文档操作记录。"""
    return {"entries": auth.audit_tail()}
