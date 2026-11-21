"""安全与编解码回归测试：纯离线，不接摄像头、不连 MySQL、不加载模型。

    pytest -q tests/
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mods import config, security  # noqa: E402


def test_hash_has_random_salt_and_verifies():
    a = security.hash_password("correct horse")
    b = security.hash_password("correct horse")
    assert security.is_hashed(a)
    assert a != b, "同一口令两次哈希必须因随机盐而不同"
    assert security.verify_password("correct horse", a)
    assert not security.verify_password("wrong password", a)


def test_legacy_plaintext_password_still_verifies():
    assert not security.is_hashed("old_secret")
    assert security.verify_password("old_secret", "old_secret")
    assert not security.verify_password("Old_secret", "old_secret")


def test_tampered_hash_is_rejected():
    stored = security.hash_password("abc12345").split("$")
    stored[3] = "0" * 64
    assert not security.verify_password("abc12345", "$".join(stored))


def test_empty_password_is_rejected():
    with pytest.raises(ValueError):
        security.hash_password("")


def test_db_config_fails_loudly_without_password(monkeypatch, tmp_path):
    monkeypatch.delenv("SECURITY_DB_PASSWORD", raising=False)
    monkeypatch.setattr(config, "DEFAULT_DOTENV", tmp_path / "does-not-exist.env")
    with pytest.raises(RuntimeError, match="SECURITY_DB_PASSWORD"):
        config.db_config()


def test_db_config_reads_dotenv(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nSECURITY_DB_PASSWORD=from_file\nSECURITY_DB_HOST=db.local\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("SECURITY_DB_PASSWORD", raising=False)
    monkeypatch.delenv("SECURITY_DB_HOST", raising=False)
    monkeypatch.setattr(config, "DEFAULT_DOTENV", env_file)
    cfg = config.db_config()
    assert cfg["password"] == "from_file"
    assert cfg["host"] == "db.local"


def test_feature_codec_roundtrip():
    np = pytest.importorskip("numpy")
    from mods import feature_codec

    vector = np.linspace(-1.0, 1.0, 512, dtype=np.float32)
    text = feature_codec.serialize_feature(vector)
    restored = feature_codec.parse_feature(text, expected_dim=512)
    assert restored is not None
    assert np.allclose(restored, vector, atol=1e-6)


def test_feature_codec_reads_legacy_formats_but_never_executes_code():
    np = pytest.importorskip("numpy")
    from mods import feature_codec

    legacy = feature_codec.parse_feature("array([0.1, 0.2, 0.3])")
    assert legacy is not None and legacy.size == 3

    payload = "__import__('os').system('echo pwned')"
    assert feature_codec.parse_feature(payload) is None, "数据库内容绝不能被当代码执行"

    assert feature_codec.parse_feature("[1.0, NaN, 3.0]") is None, (
        "含非有限值的特征串不能被数字兜底悄悄截断成短向量"
    )
    assert feature_codec.parse_feature("1.0, 2.0, 3.0") is not None
    assert feature_codec.parse_feature("") is None
