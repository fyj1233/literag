"""LiteRAG 入口：FastAPI 应用 + 静态前端托管。"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import auth
from app.api.routes import router
from app.api.settings_api import router as settings_router
from app.config import settings


@asynccontextmanager
async def lifespan(_app: FastAPI):
    auth.seed()   # 首启建用户表 + 内置演示账号（幂等）
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.include_router(router)
app.include_router(settings_router)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


# 前端若有其他静态资源一并托管
if (WEB_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIR / "assets"), name="assets")
