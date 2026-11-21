"""人脸特征比对：相似度、维度校验、阈值都集中在这一处。

原来 auto_ui.py 和 hand_ui.py 有三份几乎相同的余弦匹配代码，而且都在
"库的维度更长时静默截断前 N 维"——一条损坏或半写入的特征会看起来照样能匹配。
现在统一为严格 512 维校验：格式不对就拒绝，不猜测、不截断。
"""

from __future__ import annotations

import os

import numpy as np

from .feature_codec import EXPECTED_DIM, parse_feature, parse_template
from .template_protection import hamming_similarity, protect_feature

DEFAULT_THRESHOLD = 0.6
#: 二进制模板的阈值是「位置一致率」，不是余弦。不同人脸退化为 0.5，
#: 阈值必须明显高于 0.5 才有区分度；0.60 由 eval_face.py --protect 标定。
DEFAULT_TEMPLATE_THRESHOLD = 0.6


def _read_threshold(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return min(1.0, max(0.0, value))


def face_threshold() -> float:
    """相似度阈值可由 SECURITY_FACE_THRESHOLD 覆盖，便于做 FAR/FRR 扫描。"""
    return _read_threshold("SECURITY_FACE_THRESHOLD", DEFAULT_THRESHOLD)


def template_threshold() -> float:
    """可撤销模板的一致率阈值，由 SECURITY_TEMPLATE_THRESHOLD 覆盖。"""
    return _read_threshold("SECURITY_TEMPLATE_THRESHOLD", DEFAULT_TEMPLATE_THRESHOLD)


def cosine_similarity(a, b) -> float | None:
    """归一化后的余弦相似度；维度或数值非法时返回 None 而不是硬算。"""
    if a is None or b is None:
        return None
    try:
        left = np.asarray(a, dtype=np.float32).reshape(-1)
        right = np.asarray(b, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if left.size != EXPECTED_DIM or right.size != EXPECTED_DIM:
        return None
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return None
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm < 1e-8 or right_norm < 1e-8:
        return None
    return float(np.dot(left / left_norm, right / right_norm))


def match_feature(stored_text, feature, threshold: float | None = None):
    """返回 (相似度, 是否通过)。相似度不可算时返回 (None, False)。

    库里是 v2 可撤销模板时，用模板自带的种子把探针投影过去再算位置一致率；
    是旧版原始特征数组时走余弦。两套分数尺度不同，阈值也各自独立。
    """
    template = parse_template(stored_text)
    if template is not None:
        probe = protect_feature(feature, template.seed, bits=template.bits_len,
                                dim=template.dim)
        score = hamming_similarity(template, probe)
        if score is None:
            return None, False
        limit = template_threshold() if threshold is None else threshold
        return score, score > limit

    stored = parse_feature(stored_text, expected_dim=EXPECTED_DIM)
    score = cosine_similarity(stored, feature)
    if score is None:
        return None, False
    limit = face_threshold() if threshold is None else threshold
    return score, score > limit
