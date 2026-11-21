"""人脸比对的离线回归：不需要摄像头、数据库或模型。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mods import feature_codec, matching  # noqa: E402


def _unit_vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=512).astype(np.float32)
    return vec / np.linalg.norm(vec)


def test_same_person_passes():
    vec = _unit_vector(1)
    score, passed = matching.match_feature(feature_codec.serialize_feature(vec), vec)
    assert score is not None and score > 0.999
    assert passed is True


def test_different_person_rejected():
    owner, stranger = _unit_vector(1), _unit_vector(2)
    score, passed = matching.match_feature(
        feature_codec.serialize_feature(owner), stranger
    )
    assert score is not None and abs(score) < 0.1
    assert passed is False


def test_wrong_dimension_is_rejected_instead_of_truncated():
    """旧实现会把 600 维截断成 512 维硬比；现在必须明确拒绝。"""
    vec = _unit_vector(3)
    padded = np.concatenate([vec, vec[:88]])
    score, passed = matching.match_feature(feature_codec.serialize_feature(padded), vec)
    assert score is None and passed is False


def test_corrupt_payload_is_rejected():
    score, passed = matching.match_feature(
        "__import__('os').system('echo pwned')", _unit_vector(4)
    )
    assert score is None and passed is False


def test_threshold_is_configurable_and_clamped(monkeypatch):
    monkeypatch.setenv("SECURITY_FACE_THRESHOLD", "0.9")
    assert matching.face_threshold() == pytest.approx(0.9)
    monkeypatch.setenv("SECURITY_FACE_THRESHOLD", "2.5")
    assert matching.face_threshold() == 1.0
    monkeypatch.setenv("SECURITY_FACE_THRESHOLD", "not-a-number")
    assert matching.face_threshold() == matching.DEFAULT_THRESHOLD


def test_explicit_threshold_overrides_env(monkeypatch):
    monkeypatch.setenv("SECURITY_FACE_THRESHOLD", "0.99")
    vec = _unit_vector(5)
    other = _unit_vector(6)
    text = feature_codec.serialize_feature(vec)
    assert matching.match_feature(text, vec, threshold=0.1)[1] is True
    assert matching.match_feature(text, other, threshold=0.99)[1] is False
