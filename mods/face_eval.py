"""人脸验证指标：FAR/FRR 阈值扫描、EER 与 ROC AUC（纯 NumPy）。

评估口径统一在这里，避免 CLI 和测试各写一套：
  - 相似度 > 阈值判为"同一人"；
  - FAR = 冒名对中被放行的比例，FRR = 本人对中被拒绝的比例；
  - EER 取 FAR 与 FRR 最接近的点（0.01 步长网格），同时返回该点阈值。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_THRESHOLDS = np.round(np.arange(0.0, 1.0001, 0.01), 2)


@dataclass(frozen=True)
class SweepResult:
    thresholds: np.ndarray
    far: np.ndarray
    frr: np.ndarray
    eer: float
    eer_threshold: float
    auc: float

    @property
    def tpr(self) -> np.ndarray:
        return 1.0 - self.frr

    def at(self, threshold: float) -> dict:
        """取指定阈值下的 FAR/FRR；用于固定线上阈值时汇报。"""
        index = int(np.argmin(np.abs(self.thresholds - threshold)))
        return {
            "threshold": float(self.thresholds[index]),
            "far": float(self.far[index]),
            "frr": float(self.frr[index]),
        }

    def as_dict(self) -> dict:
        return {
            "eer": round(float(self.eer), 5),
            "eer_threshold": round(float(self.eer_threshold), 4),
            "auc": round(float(self.auc), 5),
            "thresholds": [round(float(t), 2) for t in self.thresholds],
            "far": [round(float(v), 5) for v in self.far],
            "frr": [round(float(v), 5) for v in self.frr],
        }


def _scores(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} 样本为空，无法计算指标")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 含非有限值")
    return arr


def roc_auc(genuine, impostor) -> float:
    """由相似度直接算 ROC AUC：正例是本人对，负例是冒名对。"""
    g = _scores(genuine, "genuine")
    i = _scores(impostor, "impostor")
    scores = np.concatenate([g, i])
    labels = np.concatenate([np.ones_like(g), np.zeros_like(i)])
    #: ROC 要按分数从高到低累积：先放进来的应该是最像的本人对
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    #: 同一分数必须整组处理，否则 ROC 会因并列值被算成锯齿
    distinct = np.where(np.diff(sorted_scores))[0]
    threshold_index = np.r_[distinct, sorted_scores.size - 1]
    tps = np.cumsum(sorted_labels)[threshold_index]
    fps = (threshold_index + 1) - tps
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    return float(np.trapezoid(tpr, fpr))


def threshold_sweep(genuine, impostor, thresholds=None) -> SweepResult:
    """在阈值网格上扫描，返回 FAR/FRR 曲线、EER 与 AUC。"""
    g = _scores(genuine, "genuine")
    i = _scores(impostor, "impostor")
    grid = DEFAULT_THRESHOLDS if thresholds is None else np.asarray(thresholds, dtype=np.float64)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("thresholds 必须是一维非空数组")

    far = (i[None, :] > grid[:, None]).mean(axis=1)
    frr = (g[None, :] <= grid[:, None]).mean(axis=1)
    gap = np.abs(far - frr)
    index = int(np.argmin(gap))
    eer = float((far[index] + frr[index]) / 2.0)

    return SweepResult(
        thresholds=grid,
        far=far,
        frr=frr,
        eer=eer,
        eer_threshold=float(grid[index]),
        auc=roc_auc(g, i),
    )


def verification_summary(genuine, impostor, threshold: float | None = None) -> dict:
    """一次性给出评估报告需要的指标集合。"""
    sweep = threshold_sweep(genuine, impostor)
    summary = sweep.as_dict()
    summary["genuine_pairs"] = int(np.size(genuine))
    summary["impostor_pairs"] = int(np.size(impostor))
    if threshold is not None:
        summary["at_threshold"] = sweep.at(threshold)
    return summary
