"""配置读取：数据库凭据一律来自环境变量或本地 .env，不写进源码。

.env 只放在本机、已被 .gitignore 排除；仓库里只提交 .env.example 模板。
环境变量优先级高于 .env，方便 CI / 容器里覆盖。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 打包后 __file__ 在临时解压目录里，.env 应放在 exe 同目录才找得到。
DEFAULT_DOTENV = (
    Path(sys.executable).resolve().parent / ".env"
    if getattr(sys, "frozen", False)
    else ROOT / ".env"
)


def load_dotenv(path: Path | None = None) -> None:
    """极简 .env 读取器，避免为四个变量引入 python-dotenv 依赖。"""
    file = Path(path) if path else DEFAULT_DOTENV
    if not file.exists():
        return
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def db_config() -> dict:
    """返回 pymysql.connect 所需参数；缺口令时快速失败并给出可操作提示。"""
    load_dotenv()
    config = {
        "host": os.getenv("SECURITY_DB_HOST", "localhost"),
        "user": os.getenv("SECURITY_DB_USER", "root"),
        "password": os.getenv("SECURITY_DB_PASSWORD"),
        "database": os.getenv("SECURITY_DB_NAME", "Security_System_Users_DataBase"),
        "charset": "utf8mb4",
    }
    if not config["password"]:
        raise RuntimeError(
            "缺少数据库口令：请把 .env.example 复制为 .env 并填写 "
            "SECURITY_DB_PASSWORD（或在环境变量中设置）"
        )
    return config
