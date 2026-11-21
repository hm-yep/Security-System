"""人脸特征向量的编解码：只解析数据，绝不执行数据。

旧版在 auto_ui.py 里留了 eval(feat_str) 作为最后兜底——feat_str 来自数据库，
等价于把数据库内容当代码执行。这里用 json / literal_eval / 数值扫描三条
只读路径覆盖历史格式，并统一做有限值与维度校验。
"""

from __future__ import annotations

import ast
import json
import re

import numpy as np

_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
EXPECTED_DIM = 512


def _as_vector(value) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _from_literal(text: str) -> np.ndarray | None:
    """兼容 Python 字面量以及 numpy 的 array([...]) / repr 文本。"""
    cleaned = text.strip()
    cleaned = re.sub(r"^np\.|^array\(", "", cleaned)
    cleaned = cleaned.replace("array(", "").replace("dtype=float32", "")
    cleaned = cleaned.strip().rstrip(")")
    try:
        return _as_vector(ast.literal_eval(cleaned))
    except (ValueError, SyntaxError, TypeError):
        return None


def _from_numbers(text: str) -> np.ndarray | None:
    """最后的兜底只接受"纯数字 + 分隔符"的文本。

    如果把任意字符串都拿去提数字，一段损坏的 JSON（含 NaN）或一段代码
    会被悄悄截断成一条"看起来合法"的短向量，比解析失败更危险。
    """
    if not re.fullmatch(r"[\s,;\[\]()+\-.0-9eE]+", text):
        return None
    numbers = _NUMBER_RE.findall(text)
    if not numbers:
        return None
    return _as_vector([float(n) for n in numbers])


def serialize_feature(feature) -> str:
    """注册时把 512 维特征写成 JSON 数组，替换散落各处的 json.dumps。"""
    arr = _as_vector(feature)
    if arr is None:
        raise ValueError("特征向量为空或含非有限值，拒绝入库")
    return json.dumps(arr.tolist())


def serialize_template(feature, bits: int = 512) -> str:
    """注册时写库：生成可撤销模板（v2 JSON），而不是原始 512 维特征。"""
    from .template_protection import enroll_template

    return enroll_template(feature, bits=bits)


def parse_template(text):
    """解析 v2 可撤销模板；不是 v2 格式返回 None，交给 parse_feature 走旧路径。"""
    from .template_protection import decode_template

    return decode_template(text)


def parse_feature(text, expected_dim: int | None = None) -> np.ndarray | None:
    """把库里的特征字符串还原成 float32 向量；格式不认识就返回 None。"""
    if text is None:
        return None
    if not isinstance(text, str):
        arr = _as_vector(text)
    else:
        source = text.strip()
        if not source:
            return None
        arr = None
        try:
            arr = _as_vector(json.loads(source))
        except (ValueError, TypeError):
            arr = None
        if arr is None:
            arr = _from_literal(source)
        if arr is None:
            arr = _from_numbers(source)
    if arr is None:
        return None
    if expected_dim is not None and arr.size != expected_dim:
        return None
    return arr
