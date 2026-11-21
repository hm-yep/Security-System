"""可撤销模板的离线回归：吊销、不可关联、编码往返与旧数据兼容。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mods import feature_codec, matching, template_protection as tp  # noqa: E402


def _unit_vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=tp.DEFAULT_DIM).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _near(base: np.ndarray, cosine: float = 0.8) -> np.ndarray:
    """构造一个与 base 余弦相似度约为 cosine 的探针，模拟同一人的另一帧。"""
    rng = np.random.default_rng(99)
    noise = rng.normal(size=base.size).astype(np.float32)
    noise /= np.linalg.norm(noise)
    scale = float(np.sqrt(max(0.0, 1.0 / cosine ** 2 - 1.0)))
    vec = base + scale * noise
    return vec / np.linalg.norm(vec)


SEED_A = b"\x11" * tp.SEED_BYTES
SEED_B = b"\x22" * tp.SEED_BYTES


def test_same_feature_same_seed_is_stable():
    face = _unit_vector(1)
    first = tp.protect_feature(face, SEED_A)
    second = tp.protect_feature(face, SEED_A)
    assert first is not None and second is not None
    assert np.array_equal(first.bits, second.bits)
    assert tp.hamming_similarity(first, second) == 1.0


def test_new_seed_revokes_old_template():
    """吊销的核心证据：同一个脸换种子后，模板一致率退化为随机 ≈0.5。"""
    face = _unit_vector(2)
    old = tp.protect_feature(face, SEED_A)
    renewed = tp.protect_feature(face, SEED_B)
    score = tp.hamming_similarity(old, renewed)
    assert score is not None
    assert 0.40 < score < 0.60


def test_cross_system_templates_are_unlinkable():
    """两套系统不同种子：同一个人也无法把两边模板关联起来。"""
    face = _unit_vector(3)
    system_a = tp.protect_feature(face, SEED_A)
    system_b = tp.protect_feature(face, SEED_B)
    unrelated = tp.protect_feature(_unit_vector(4), SEED_A)
    linked = tp.hamming_similarity(system_a, system_b)
    impostor = tp.hamming_similarity(system_a, unrelated)
    assert linked is not None and impostor is not None
    assert abs(linked - impostor) < 0.08


def test_nearby_features_stay_similar():
    face = _unit_vector(5)
    enrolled = tp.protect_feature(face, SEED_A)
    probe = tp.protect_feature(_near(face), SEED_A)
    score = tp.hamming_similarity(enrolled, probe)
    assert score is not None and score > 0.7


def test_fuse_templates_uses_majority_vote():
    rng = np.random.default_rng(7)
    bits = rng.integers(0, 2, size=64).astype(np.uint8)
    one = tp.ProtectedTemplate(SEED_A, bits.copy())
    two = tp.ProtectedTemplate(SEED_A, bits.copy())
    three = tp.ProtectedTemplate(SEED_A, np.zeros(64, dtype=np.uint8))
    fused = tp.fuse_templates([one, two, three])
    assert fused is not None
    assert np.array_equal(fused.bits, bits)


def test_fuse_rejects_mixed_seeds():
    one = tp.ProtectedTemplate(SEED_A, np.zeros(16, dtype=np.uint8))
    two = tp.ProtectedTemplate(SEED_B, np.zeros(16, dtype=np.uint8))
    assert tp.fuse_templates([one, two]) is None


def test_encode_decode_roundtrip():
    face = _unit_vector(6)
    template = tp.protect_feature(face, SEED_A)
    text = tp.encode_template(template)
    restored = tp.decode_template(text)
    assert restored is not None
    assert restored.seed == template.seed
    assert restored.dim == template.dim
    assert np.array_equal(restored.bits, template.bits)
    assert tp.hamming_similarity(template, restored) == 1.0


def test_decode_rejects_malformed_payloads():
    assert tp.decode_template("[0.1, 0.2]") is None
    assert tp.decode_template('{"v":2,"seed":"zz","n":8,"dim":512,"bits":"AA=="}') is None
    assert tp.decode_template('{"v":2,"seed":"' + SEED_A.hex() + '","n":8,"dim":512,"bits":""}') is None
    assert tp.decode_template('{"v":2,"seed":"' + SEED_A.hex() + '","n":512,"dim":128,"bits":"AA=="}') is None


def test_matching_accepts_v2_genuine_and_rejects_impostor():
    owner = _unit_vector(8)
    text = feature_codec.serialize_template(owner)
    assert feature_codec.parse_template(text) is not None

    score, passed = matching.match_feature(text, owner)
    assert score is not None and score > 0.99 and passed is True

    score, passed = matching.match_feature(text, _unit_vector(9))
    assert score is not None and not passed


def test_legacy_raw_feature_still_uses_cosine():
    """旧库里的原始 512 维 JSON 数组不能因为升级就全部失效。"""
    owner = _unit_vector(10)
    text = feature_codec.serialize_feature(owner)
    assert feature_codec.parse_template(text) is None
    score, passed = matching.match_feature(text, owner)
    assert score is not None and score > 0.999 and passed is True


def test_template_threshold_is_configurable(monkeypatch):
    monkeypatch.setenv("SECURITY_TEMPLATE_THRESHOLD", "0.75")
    assert matching.template_threshold() == pytest.approx(0.75)
    monkeypatch.setenv("SECURITY_TEMPLATE_THRESHOLD", "3")
    assert matching.template_threshold() == 1.0
    monkeypatch.setenv("SECURITY_TEMPLATE_THRESHOLD", "bad")
    assert matching.template_threshold() == matching.DEFAULT_TEMPLATE_THRESHOLD


def test_enroll_template_uses_fresh_seed_each_time():
    """同一张脸两次注册必须产生不同模板，否则吊销没有意义。"""
    face = _unit_vector(11)
    first = tp.decode_template(tp.enroll_template(face))
    second = tp.decode_template(tp.enroll_template(face))
    assert first is not None and second is not None
    assert first.seed != second.seed
    assert tp.hamming_similarity(first, second) < 0.6
