@echo off
setlocal
cd /d "%~dp0"
title LiteRAG 知识库问答
set "PY=.venv\Scripts\python.exe"
set "PORT=8000"

echo.
echo  ==============================================
echo    LiteRAG 知识库问答 - 一键启动
echo  ==============================================
echo.

rem ---------- 已在运行则直接打开页面 ----------
"%PY%" scripts\health_probe.py %PORT% >nul 2>nul
if not errorlevel 1 goto :already_running

rem ---------- 1. 运行环境 ----------
echo [1/4] 检查运行环境...
if not exist "%PY%" goto :build_env
"%PY%" -c "print(1)" >nul 2>nul
if errorlevel 1 goto :rebuild_env
goto :env_ok

:build_env
echo       首次运行：正在创建 Python 环境（约需几分钟，取决于网速）...
%WINDIR%\System32\where.exe uv >nul 2>nul
if not errorlevel 1 (
    uv venv --python 3.12
    uv pip install --python "%PY%" -r requirements.txt
    goto :check_env
)
py -3.12 -m venv .venv 2>nul
if not exist "%PY%" py -3 -m venv .venv 2>nul
if not exist "%PY%" python -m venv .venv 2>nul
if exist "%PY%" "%PY%" -m pip install -r requirements.txt
goto :check_env

:rebuild_env
echo       检测到环境损坏（如项目被移动过），正在自动重建...
rmdir /s /q .venv
goto :build_env

:check_env
if not exist "%PY%" goto :env_fail
"%PY%" -c "import fastapi" >nul 2>nul
if errorlevel 1 "%PY%" -m pip install -r requirements.txt

:env_ok
echo       [OK] 运行环境就绪

rem ---------- 2. 本地模型 ----------
if exist "models\bge-small-zh-v1.5\model.safetensors" goto :models_ok
echo [2/4] 首次运行：正在下载本地模型（约 1.2GB，走国内镜像，请耐心等待）...
"%PY%" scripts\download_models.py
if not exist "models\bge-small-zh-v1.5\model.safetensors" goto :model_fail
:models_ok
echo       [OK] 本地模型就绪

rem ---------- 3. 知识库索引 ----------
"%PY%" -c "import sys; sys.path.insert(0, '.'); from app.rag.pipeline import get_pipeline; sys.exit(0 if get_pipeline().vector_store.count() else 1)" >nul 2>nul
if errorlevel 1 (
    echo [3/4] 首次运行：正在灌入内置演示文档...
    "%PY%" scripts\ingest.py
)
echo       [OK] 知识库索引就绪

rem ---------- 4. 启动服务 ----------
echo [4/4] 正在启动服务...
echo.
echo  ============================================================
echo    启动成功后浏览器将自动打开 http://127.0.0.1:%PORT%
echo    首次提问需要加载模型（约 30 秒），之后每次约 2 秒
echo    关闭本窗口 或 按 Ctrl+C 即可停止服务
echo  ============================================================
echo.
if not defined LITERAG_NO_OPEN start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 8; Start-Process 'http://127.0.0.1:%PORT%'"
title LiteRAG 服务运行中（端口 %PORT%）- 关闭本窗口即停止
"%PY%" -m uvicorn app.main:app --host 127.0.0.1 --port %PORT%
echo.
echo 服务已停止。
pause
exit /b 0

:already_running
echo 检测到 LiteRAG 服务已在运行，直接为你打开页面...
if not defined LITERAG_NO_OPEN start "" http://127.0.0.1:%PORT%
%WINDIR%\System32\ping.exe -n 4 127.0.0.1 >nul
exit /b 0

:env_fail
echo [错误] 运行环境创建失败：请先安装 Python 3.12 或 uv，然后重新运行本脚本。
pause
exit /b 1

:model_fail
echo [错误] 模型下载失败：请检查网络后重新运行本脚本。
pause
exit /b 1
