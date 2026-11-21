"""把 verify_claims.py 的逐条验收包装成 pytest。

真机硬件相关检查（键鼠钩子、摄像头/模型）在缺依赖时自动跳过；
本机跑全量：`SECURITY_TEST_HOOK=1 pytest -q tests/`
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import verify_claims as claims  # noqa: E402


def _run(check_group) -> None:
    claims.fails.clear()
    check_group()
    assert not claims.fails, f"验收未通过: {claims.fails}"


def _require_vision_stack() -> None:
    pytest.importorskip("PyQt5", reason="缺少 PyQt5")
    pytest.importorskip("torch", reason="缺少 torch")
    pytest.importorskip("mediapipe", reason="缺少 mediapipe")
    pytest.importorskip("ultralytics", reason="缺少 ultralytics")


def test_liveness_state_machine():
    _require_vision_stack()
    _run(claims.liveness_checks)


def test_lock_screen_requires_liveness():
    _require_vision_stack()
    _run(claims.lock_liveness_checks)


def test_system_guards_static():
    _require_vision_stack()
    _run(claims.system_checks)


@pytest.mark.skipif(not (ROOT / "yolov8n.pt").exists(), reason="缺少 yolov8n.pt")
def test_yolo_multi_person():
    pytest.importorskip("ultralytics", reason="缺少 ultralytics")
    _run(claims.yolo_checks)


def test_audit_and_startup():
    _run(claims.audit_checks)
