@echo off
setlocal
set "PORT=8000"
set FOUND=0

for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":%PORT% " ^| findstr "LISTENING"') do (
    %WINDIR%\System32\taskkill.exe /F /PID %%a >nul 2>nul && set FOUND=1
)

if "%FOUND%"=="1" (
    echo LiteRAG 服务已停止。
) else (
    echo 未发现正在运行的 LiteRAG 服务。
)
%WINDIR%\System32\ping.exe -n 3 127.0.0.1 >nul
