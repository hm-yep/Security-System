"""口令哈希与旧数据平滑迁移。

只用标准库：PBKDF2-HMAC-SHA256 + 随机盐 + 常数时间比较。
存库格式：pbkdf2_sha256$<迭代次数>$<salt_hex>$<hash_hex>

历史库里的 password 列是明文，直接切哈希会让老用户全部登不进去。
因此 verify_password 同时认明文，登录成功后由数据层把该行自动升级成哈希。
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

ALGO = "pbkdf2_sha256"
ITERATIONS = 200_000
SALT_BYTES = 16
PREFIX = f"{ALGO}$"


def hash_password(plain: str) -> str:
    if not isinstance(plain, str) or not plain:
        raise ValueError("密码不能为空")
    salt = secrets.token_bytes(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"), salt, ITERATIONS)
    return f"{ALGO}${ITERATIONS}${salt.hex()}${digest.hex()}"


def is_hashed(stored: object) -> bool:
    if not isinstance(stored, str) or not stored.startswith(PREFIX):
        return False
    parts = stored.split("$")
    return (
        len(parts) == 4
        and parts[1].isdigit()
        and len(parts[2]) == SALT_BYTES * 2
        and len(parts[3]) == 64
    )


def verify_password(plain: str, stored: object) -> bool:
    """校验口令。stored 为历史明文时走兼容分支，由调用方负责升级。"""
    if plain is None or stored is None:
        return False
    plain = str(plain)
    stored = str(stored)
    if not is_hashed(stored):
        return hmac.compare_digest(plain, stored)
    _, iterations, salt_hex, digest_hex = stored.split("$")
    candidate = hashlib.pbkdf2_hmac(
        "sha256", plain.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
    )
    return hmac.compare_digest(candidate.hex(), digest_hex)
