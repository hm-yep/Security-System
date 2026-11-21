"""验证指标的离线回归：FAR/FRR、EER、AUC 的数值正确性。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mods.face_eval import (  # noqa: E402
    roc_auc,
    threshold_sweep,
    verification_summary,
)


def test_perfect_separation():
    sweep = threshold_sweep([0.9, 0.8], [0.7, 0.2])
    assert sweep.eer == 0.0
    assert sweep.auc == pytest.approx(1.0)
    assert sweep.eer_threshold == pytest.approx(0.7, abs=0.011)


def test_identical_score_is_chance_level():
    sweep = threshold_sweep([0.6], [0.6])
    assert sweep.eer == pytest.approx(0.5, abs=0.01)
    assert sweep.auc == pytest.approx(0.5)


def test_far_decreases_and_frr_increases_with_threshold():
    rng = np.random.default_rng(0)
    genuine = rng.normal(0.8, 0.05, 200)
    impostor = rng.normal(0.5, 0.05, 400)
    sweep = threshold_sweep(genuine, impostor)
    assert np.all(np.diff(sweep.far) <= 1e-12)
    assert np.all(np.diff(sweep.frr) >= -1e-12)
    assert sweep.auc > 0.99
    assert sweep.eer < 0.01


def test_at_threshold_counts_are_correct():
    # t=0.75 时两对正例都被接受，两对负例都被拒绝
    sweep = threshold_sweep([0.9, 0.8], [0.7, 0.2])
    point = sweep.at(0.75)
    assert point["threshold"] == pytest.approx(0.75, abs=0.011)
    assert point["far"] == 0.0
    assert point["frr"] == 0.0


def test_auc_is_symmetric_for_swapped_distributions():
    genuine = np.linspace(0.55, 0.95, 50)
    impostor = np.linspace(0.35, 0.75, 50)
    auc = roc_auc(genuine, impostor)
    assert auc > 0.5
    assert roc_auc(impostor, genuine) == pytest.approx(1.0 - auc, abs=1e-9)


def test_summary_reports_pair_counts():
    summary = verification_summary([0.9, 0.8, 0.7], [0.6, 0.5], threshold=0.6)
    assert summary["genuine_pairs"] == 3
    assert summary["impostor_pairs"] == 2
    assert summary["at_threshold"]["threshold"] == pytest.approx(0.6, abs=1e-9)
    assert len(summary["thresholds"]) == len(summary["far"]) == len(summary["frr"])


def test_empty_scores_rejected():
    with pytest.raises(ValueError):
        threshold_sweep([], [0.5])
    with pytest.raises(ValueError):
        roc_auc([0.5], [])
