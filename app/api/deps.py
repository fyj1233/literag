"""FastAPI 认证依赖：Bearer token 解析与角色门禁（HTTP 层专用）。"""
from fastapi import Depends, HTTPException, Header

from app.config import ROLE_LEVELS


def current_user(authorization: str = Header(default="")) -> dict:
    """解析 Bearer JWT → {username, role, allowed(可见密级)}。无效则 401。"""
    from app import auth   # 函数内导入，避免循环依赖

    token = authorization[7:] if authorization.startswith("Bearer ") else ""
    claims = auth.verify_token(token)
    if not claims:
        raise HTTPException(401, "未登录或登录已过期")
    role = claims.get("role", "guest")
    return {
        "username": claims.get("sub", ""),
        "role": role,
        "allowed": ROLE_LEVELS.get(role, ROLE_LEVELS["guest"]),  # 未知角色按最低权限
    }


def require_role(*roles: str):
    """角色门禁依赖工厂：角色不满足直接 403。"""
    def dep(user: dict = Depends(current_user)) -> dict:
        if user["role"] not in roles:
            raise HTTPException(403, f"需要 {'/'.join(roles)} 权限")
        return user
    return dep
