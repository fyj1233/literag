"""探测 LiteRAG 服务是否在运行：退出码 0=运行中，1=未运行。供一键启动脚本调用。"""
import sys
import urllib.request

try:
    urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/api/health", timeout=1.5)
    sys.exit(0)
except Exception:
    sys.exit(1)
