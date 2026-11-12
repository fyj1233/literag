"""全局配置：所有可调参数集中在 .env，一处生效。"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------- 分级权限（RBAC 演示版）----------
# 角色 → 可见密级。检索层按此过滤：无权限的内容在召回阶段就被排除（默认拒绝）。
# 演示用明文角色；生产化时把"角色从哪来"换成 JWT 解析即可，过滤层代码不变。
ROLE_LEVELS: dict[str, list[str]] = {
    "boss": ["L1", "L2", "L3", "L4"],
    "employee": ["L1", "L2"],
    "guest": ["L1"],
}
DEFAULT_VISIBILITY = "L1"   # 未显式指定密级的文档默认公开


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- 应用 ----------
    app_name: str = "LiteRAG"

    # ---------- 路径 ----------
    data_dir: Path = BASE_DIR / "data"
    upload_dir: Path = BASE_DIR / "uploads"

    # ---------- LLM ----------
    # mock = 离线演示（抽取式回答）；openai = 任意 OpenAI 兼容接口
    llm_provider: str = "mock"
    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024

    # ---------- Embedding ----------
    # local | siliconflow | hash
    embedding_provider: str = "local"
    local_embedding_model: str = "BAAI/bge-small-zh-v1.5"
    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"

    # ---------- Reranker ----------
    # local | siliconflow | none
    rerank_provider: str = "local"
    local_reranker_model: str = "BAAI/bge-reranker-base"

    # ---------- 切分 ----------
    chunk_size: int = 500
    chunk_overlap: int = 80

    # ---------- 检索 ----------
    vector_top_k: int = 10
    bm25_top_k: int = 10
    rrf_k: int = 60
    candidate_k: int = 20
    final_top_k: int = 5

    # ---------- 功能开关 ----------
    enable_hybrid: bool = True
    enable_rerank: bool = True
    enable_query_rewrite: bool = True
    enable_multi_query: bool = True
    enable_hyde: bool = False
    # CRAG-lite：精排 Top1 置信分低于阈值时自动纠正重查一次（mode=full 且开启精排时生效）
    enable_corrective: bool = False
    corrective_threshold: float = 0.35
    # Parent-Child 分块：小块（chunk）检索、父块（章节全文，截断）生成
    enable_parent_child: bool = False
    parent_max_chars: int = 1200

    # ---------- 视觉识图（图片理解入库） ----------
    # 摄入时抽取文档内嵌图片 → 识图模型生成中文描述 → 作为特殊 chunk 入库
    vision_enabled: bool = True
    vision_base_url: str = ""    # 三项留空 = 直接使用主 LLM 端点/密钥/模型（要求该模型支持识图）
    vision_api_key: str = ""
    vision_model: str = ""
    vision_max_images: int = 5   # 每份文档最多识图数量（控制成本与耗时）

    # ---------- 模型下载镜像 ----------
    hf_endpoint: str = "https://hf-mirror.com"

    # ---------- 认证 ----------
    auth_secret: str = "dev-secret-literag-change-in-prod"   # JWT 签名密钥，生产必须在 .env 里改掉

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def bm25_dir(self) -> Path:
        return self.data_dir / "bm25"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.chroma_dir, self.bm25_dir, self.upload_dir):
            p.mkdir(parents=True, exist_ok=True)


def resolve_model_path(name: str) -> str:
    """模型名解析：若项目 models/ 下存在对应的本地模型目录则用绝对路径（离线加载），
    否则原样返回，当作 HuggingFace repo id（自动下载到 HF_HOME 缓存）。"""
    p = Path(name)
    if not p.is_absolute():
        p = BASE_DIR / p
    return str(p) if (p / "config.json").exists() else name


# ---------- .env 运行时持久化（设置面板用） ----------
ENV_PATH = BASE_DIR / ".env"


def persist_env(updates: dict[str, str]) -> None:
    """把键值更新写回 .env：保留原有行与注释，存在的键原位替换，新键追加到末尾。

    原子写入（先写临时文件再替换），供设置面板热更新后持久化。
    """
    import re

    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    remaining = dict(updates)
    out: list[str] = []
    for line in lines:
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=", line)
        if m and m.group(1) in remaining:
            out.append(f"{m.group(1)}={remaining.pop(m.group(1))}")
        else:
            out.append(line)
    for k, v in remaining.items():
        out.append(f"{k}={v}")
    tmp = ENV_PATH.with_suffix(".env.tmp")
    tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
    tmp.replace(ENV_PATH)


def get_settings() -> Settings:
    """读取配置；同时把 HF 镜像与缓存目录写入环境变量（须在 transformers 导入前生效）。"""
    s = Settings()
    if s.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", s.hf_endpoint)
    # 模型缓存也放进项目目录，保证整个项目自包含、不污染 C 盘
    os.environ.setdefault("HF_HOME", str(s.data_dir.parent / "models" / "hf-home"))
    s.ensure_dirs()
    return s


settings = get_settings()
