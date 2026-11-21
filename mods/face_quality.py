"""人脸质量门控与多帧特征融合。

单帧提特征有两个老问题：
  1. 模糊、逆光、侧脸、离镜头太远的帧照样送进 ResNet，注册进去的是脏特征；
  2. 一帧噪声直接决定比对结果，相似度会在阈值附近来回抖。

这里用两件只依赖 OpenCV + NumPy 的小事解决：先按清晰度/亮度/对比度/姿态/
人脸占比给每帧打分，只有通过的帧才允许提特征；再把多帧 L2 归一化后的特征取
中位数融合，压掉单帧噪声和偶发坏帧。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

EXPECTED_DIM = 512

#: 硬门槛：任一不满足直接丢弃该帧，不参与融合。
#: 评分统一在 MTCNN 对齐后的 160×160 人脸上做，因此阈值跨数据源可比；
#: 低分辨率人脸被放大后 Laplacian 方差天然偏低，阈值取 15 兼顾召回。
MIN_SHARPNESS = 15.0          # Laplacian 方差，低于此基本是糊的
MIN_CONTRAST = 15.0           # 灰度标准差
BRIGHT_RANGE = (45.0, 215.0)  # 过暗 / 过曝
MAX_YAW = 30.0                # 与活体判据同一套 0~60 标尺
MAX_ROLL = 22.0               # 双眼连线倾角（度）
MIN_FACE_RATIO = 0.015        # 人脸框面积 / 画面面积；640×480 下约 68×68 像素
MIN_SCORE = 0.45              # 综合分门槛


@dataclass(frozen=True)
class QualityReport:
    """单帧人脸的质量评估结果，字段保留原始值便于日志与调参。"""

    score: float
    sharpness: float
    brightness: float
    contrast: float
    yaw: float
    roll: float
    face_ratio: float
    issues: tuple = ()

    @property
    def passed(self) -> bool:
        return self.score >= MIN_SCORE and not self.issues


def _as_gray(face) -> np.ndarray | None:
    """统一转灰度；传入的既可能是灰度图，也可能是 RGB/BGR 小块。"""
    arr = np.asarray(face)
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        return None
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def pose_from_landmarks(landmarks) -> tuple[float, float]:
    """用 MTCNN 的 5 点关键点估计 yaw/roll（度）。

    yaw 沿用项目里活体判据的近似：鼻尖相对双眼中点的水平偏移 / 眼距 × 60。
    数值本身不是物理角度，但同一套标尺下阈值可复现、可测试。
    """
    try:
        pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
    except (TypeError, ValueError):
        return 0.0, 0.0
    if pts.shape[0] < 3:
        return 0.0, 0.0
    left_eye, right_eye, nose = pts[0], pts[1], pts[2]
    eye_dx = float(right_eye[0] - left_eye[0])
    eye_dy = float(right_eye[1] - left_eye[1])
    eye_dist = max(float(np.hypot(eye_dx, eye_dy)), 1e-6)
    eye_mid_x = (float(left_eye[0]) + float(right_eye[0])) / 2.0
    yaw = abs(float(nose[0]) - eye_mid_x) / eye_dist * 60.0
    roll = abs(float(np.degrees(np.arctan2(eye_dy, eye_dx))))
    return min(yaw, 90.0), min(roll, 90.0)


def assess_face(face_gray, box=None, landmarks=None, frame_shape=None) -> QualityReport:
    """给一帧人脸打分。

    face_gray: 人脸区域灰度图（uint8）或 RGB/BGR 小块。
    box: (x1, y1, x2, y2) 像素坐标，用于计算人脸占比。
    landmarks: MTCNN 的 5 点关键点。
    frame_shape: 原画面 (h, w)，缺省时人脸占比记为 1.0（只做局部评估）。
    """
    gray = _as_gray(face_gray)
    if gray is None:
        return QualityReport(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ("无有效人脸区域",))

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())

    yaw, roll = pose_from_landmarks(landmarks) if landmarks is not None else (0.0, 0.0)

    face_ratio = 1.0
    if box is not None and frame_shape is not None:
        try:
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            height, width = float(frame_shape[0]), float(frame_shape[1])
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if width > 0 and height > 0:
                face_ratio = area / (width * height)
        except (TypeError, ValueError, IndexError):
            face_ratio = 0.0

    sharp_score = min(1.0, sharpness / 80.0)
    bright_score = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
    contrast_score = min(1.0, contrast / 40.0)
    pose_score = max(0.0, 1.0 - max(yaw / 45.0, roll / 35.0))
    size_score = min(1.0, face_ratio / 0.08)
    score = (
        0.35 * sharp_score
        + 0.15 * bright_score
        + 0.15 * contrast_score
        + 0.20 * pose_score
        + 0.15 * size_score
    )

    issues = []
    if sharpness < MIN_SHARPNESS:
        issues.append("画面模糊")
    if brightness < BRIGHT_RANGE[0]:
        issues.append("光线过暗")
    elif brightness > BRIGHT_RANGE[1]:
        issues.append("光线过曝")
    if contrast < MIN_CONTRAST:
        issues.append("对比度不足")
    if yaw > MAX_YAW:
        issues.append("侧脸角度过大")
    if roll > MAX_ROLL:
        issues.append("头部倾斜")
    if face_ratio < MIN_FACE_RATIO:
        issues.append("离镜头太远")

    return QualityReport(
        score=round(float(score), 4),
        sharpness=round(sharpness, 2),
        brightness=round(brightness, 2),
        contrast=round(contrast, 2),
        yaw=round(yaw, 2),
        roll=round(roll, 2),
        face_ratio=round(face_ratio, 5),
        issues=tuple(issues),
    )


def fuse_features(features, expected_dim: int = EXPECTED_DIM) -> np.ndarray | None:
    """多帧特征融合：逐条 L2 归一化后取中位数，再归一化输出。

    用中位数而不是均值：均值会被一帧偶发脏特征拉偏，中位数在样本数过半正确时
    几乎不受影响。维度不符、含非有限值或零向量的样本直接跳过。
    """
    vectors = []
    for feature in features or ():
        try:
            arr = np.asarray(feature, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            continue
        if arr.size != expected_dim or not np.all(np.isfinite(arr)):
            continue
        norm = float(np.linalg.norm(arr))
        if norm < 1e-8:
            continue
        vectors.append(arr / norm)
    if not vectors:
        return None
    fused = np.median(np.stack(vectors, axis=0), axis=0)
    norm = float(np.linalg.norm(fused))
    if norm < 1e-8:
        return None
    return (fused / norm).astype(np.float32)
