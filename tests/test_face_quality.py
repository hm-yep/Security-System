"""人脸质量门控与多帧融合的离线回归：不接摄像头、不加载模型、不连数据库。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mods.face_quality import (  # noqa: E402
    EXPECTED_DIM,
    assess_face,
    fuse_features,
    pose_from_landmarks,
)


def textured_face(size=160, contrast=200, offset=30) -> np.ndarray:
    """高频棋盘格：清晰度和对比度都足够，代表一帧可用的人脸。"""
    yy, xx = np.mgrid[0:size, 0:size]
    checker = (((xx // 8) + (yy // 8)) % 2).astype(np.uint8)
    return (offset + checker * contrast).astype(np.uint8)


def centered_landmarks():
    """5 点关键点：双眼中点与鼻尖水平对齐，即正脸。"""
    return np.array(
        [[40.0, 60.0], [120.0, 60.0], [80.0, 90.0], [55.0, 120.0], [105.0, 120.0]],
        dtype=np.float32,
    )


def test_sharp_frontal_face_passes():
    face = textured_face()
    report = assess_face(face, box=(0, 0, 160, 160), landmarks=centered_landmarks(),
                         frame_shape=(480, 640))
    assert report.passed, report.issues
    assert report.sharpness > 100
    assert report.yaw < 1.0

def test_blurred_face_rejected():
    #: 25×25 高斯模糊模拟明显运动模糊/失焦，Laplacian 方差会掉到 10 以下
    face = cv2.GaussianBlur(textured_face(), (25, 25), 0)
    report = assess_face(face, box=(0, 0, 160, 160), landmarks=centered_landmarks(),
                         frame_shape=(480, 640))
    assert not report.passed
    assert "画面模糊" in report.issues


def test_dark_face_rejected():
    face = (textured_face().astype(np.float32) * 0.1).astype(np.uint8)
    report = assess_face(face, box=(0, 0, 160, 160), frame_shape=(480, 640))
    assert not report.passed
    assert "光线过暗" in report.issues


def test_flat_image_rejected():
    report = assess_face(np.full((160, 160), 128, dtype=np.uint8))
    assert not report.passed


def test_large_yaw_rejected():
    points = centered_landmarks()
    points[2] = (160.0, 90.0)  # 鼻尖大幅偏离双眼中点
    report = assess_face(textured_face(), box=(0, 0, 160, 160), landmarks=points,
                         frame_shape=(480, 640))
    assert report.yaw > 45
    assert not report.passed
    assert "侧脸角度过大" in report.issues


def test_small_face_ratio_rejected():
    report = assess_face(textured_face(64), box=(0, 0, 64, 64), frame_shape=(1080, 1920))
    assert report.face_ratio < 0.01
    assert "离镜头太远" in report.issues


def test_pose_from_landmarks_matches_existing_scale():
    yaw, roll = pose_from_landmarks(centered_landmarks())
    assert yaw == pytest.approx(0.0, abs=1e-4)
    assert roll == pytest.approx(0.0, abs=1e-4)

    tilted = centered_landmarks()
    tilted[0] = (40.0, 40.0)
    tilted[1] = (120.0, 80.0)
    _yaw, roll = pose_from_landmarks(tilted)
    assert roll > 20


def test_fuse_features_normalizes_and_rejects_bad_samples():
    base = np.linspace(-1.0, 1.0, EXPECTED_DIM, dtype=np.float32)
    noisy = base + np.random.default_rng(0).normal(0, 0.05, EXPECTED_DIM).astype(np.float32)

    fused = fuse_features([base, noisy])
    assert fused is not None
    assert fused.shape == (EXPECTED_DIM,)
    assert np.linalg.norm(fused) == pytest.approx(1.0, abs=1e-5)

    bad = np.full(EXPECTED_DIM, np.nan, dtype=np.float32)
    assert fuse_features([base, bad]) is not None
    assert fuse_features([bad, np.zeros(EXPECTED_DIM)]) is None
    assert fuse_features([]) is None
    assert fuse_features([np.ones(10, dtype=np.float32)]) is None


def test_fusion_median_suppresses_outlier():
    base = np.linspace(0.1, 1.0, EXPECTED_DIM, dtype=np.float32)
    outlier = -base
    fused = fuse_features([base, base, outlier])
    assert fused is not None
    similarity = float(np.dot(fused, base / np.linalg.norm(base)))
    assert similarity > 0.99
