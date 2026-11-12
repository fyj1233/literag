"""企业级认证与审计模块（纯标准库实现，零额外依赖）。

组成：
1. UserStore      —— SQLite 用户表（PBKDF2-HMAC-SHA256 密码哈希，随机盐，10 万次迭代）
2. JWT（HS256）   —— 按 RFC 7519 手写：header.payload.signature，HMAC-SHA256 签名，带过期时间
3. audit()        —— 审计日志（谁在什么时候问了什么、命中了哪些来源）

设计说明：
- 密码哈希用 PBKDF2 而非 bcrypt：标准库自带、无可移植性问题，10 万次迭代符合 OWASP 基线
- JWT 手写 HS256：结构透明可教学；生产可直接替换为 PyJWT，接口不变
- 本模块不依赖 FastAPI——HTTP 层的解析放在 api/routes.py 的依赖注入里，保持分层干净
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import sqlite3
import time

from app.config import BASE_DIR, settings

DB_PATH = BASE_DIR / "data" / "users.db"      # 用户与审计库（项目内，不进 git）
AUDIT_LIMIT = 200                             # 审计查询最多返回条数
TOKEN_TTL = 12 * 3600                         # token 有效期 12 小时

_SEED_USERS = [  # (username, password, role) —— 首次启动自动创建，README 有说明
    ("boss", "boss123", "boss"),
    ("employee", "employee123", "employee"),
    ("guest", "guest123", "guest"),
]


# ==================== 密码哈希 ====================
def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return f"{salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    salt, dk_hex = stored.split("$", 1)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return hmac.compare_digest(dk.hex(), dk_hex)


# ==================== 用户表（SQLite） ====================
def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS users ("
            "username TEXT PRIMARY KEY, pw TEXT NOT NULL, role TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS audit_log ("
            "ts REAL, username TEXT, endpoint TEXT, question TEXT, sources TEXT)"
        )
        for username, password, role in _SEED_USERS:
            conn.execute(
                "INSERT OR IGNORE INTO users(username, pw, role) VALUES (?, ?, ?)",
                (username, hash_password(password), role),
            )


def create_user(username: str, password: str, role: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO users(username, pw, role) VALUES (?, ?, ?)",
            (username, hash_password(password), role),
        )


def get_user(username: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT username, pw, role FROM users WHERE username = ?", (username,)
        ).fetchone()
    return {"username": row[0], "pw": row[1], "role": row[2]} if row else None


def seed() -> None:
    """首启建表 + 内置三个演示账号（幂等）。"""
    init_db()


# ==================== JWT（HS256，RFC 7519） ====================
def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def make_token(username: str, role: str) -> str:
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    now = int(time.time())
    payload = _b64url(json.dumps({
        "sub": username, "role": role, "iat": now, "exp": now + TOKEN_TTL,
    }).encode())
    signing_input = f"{header}.{payload}".encode()
    signature = _b64url(hmac.new(settings.auth_secret.encode(), signing_input, hashlib.sha256).digest())
    return f"{header}.{payload}.{signature}"


def verify_token(token: str) -> dict | None:
    """校验签名与有效期；合法返回 payload，否则 None。"""
    try:
        header, payload, signature = token.split(".")
        signing_input = f"{header}.{payload}".encode()
        expected = hmac.new(settings.auth_secret.encode(), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64url_decode(signature)):
            return None
        claims = json.loads(_b64url_decode(payload))
        if claims.get("exp", 0) < time.time():
            return None
        return claims
    except Exception:  # noqa: BLE001 —— 任何格式/签名问题都视为无效 token
        return None


def login(username: str, password: str) -> dict | None:
    """校验账号密码；成功返回 {token, username, role}。"""
    user = get_user(username)
    if not user or not verify_password(password, user["pw"]):
        return None
    return {
        "token": make_token(user["username"], user["role"]),
        "username": user["username"],
        "role": user["role"],
    }


# ==================== 审计日志 ====================
def audit(username: str, endpoint: str, question: str, sources: list[str]) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO audit_log(ts, username, endpoint, question, sources) VALUES (?, ?, ?, ?, ?)",
            (time.time(), username, endpoint, question[:300], ",".join(sources)),
        )


def audit_tail(limit: int = AUDIT_LIMIT) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT ts, username, endpoint, question, sources FROM audit_log "
            "ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {"ts": r[0], "username": r[1], "endpoint": r[2], "question": r[3], "sources": r[4]}
        for r in rows
    ]
