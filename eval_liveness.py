"""活体检测量化评估：不接摄像头，用注入式关键点模拟真人与三类攻击。

    python eval_liveness.py                # 默认 200 轮，写 reports/liveness.json
    python eval_liveness.py -n 50 --no-write

测试对象就是实际解锁路径里的 run_liveness，不是另写一套判据。
输出通过率 / 拦截率，包括当前实现对预录视频的真实短板——不粉饰。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from verify_claims import landmarks  # noqa: E402  复用造脸工厂，避免重复判据
from mods import face_detector as fd  # noqa: E402
from mods.face_detector import run_liveness  # noqa: E402

VIDEO_SEQUENCE = ("blink", "mouth", "pose", "motion")
CAMERA = type("FakeCam", (), {"getframe": lambda self: "frame"})()


@dataclass(frozen=True)
class Outcome:
    name: str
    description: str
    rounds: int
    passed: int
    elapsed: float

    @property
    def pass_rate(self) -> float:
        return self.passed / self.rounds if self.rounds else 0.0


def _prompt_factory(state: dict):
    def prompt(text: str) -> None:
        for challenge in fd.LIVENESS_CHALLENGES:
            if challenge.prompt == text:
                state["key"] = challenge.key
                state["calls"] = 0
                return
    return prompt


def _live_provider(state: dict):
    def provider(_frame):
        key = state["key"]
        state["calls"] += 1
        shift = 0.02 if key == "motion" and state["calls"] > 1 else 0.0
        return landmarks(blink=key == "blink", mouth=key == "mouth", shift=shift)
    return provider


def _photo_provider(_frame):
    return landmarks()


def _video_provider(state: dict):
    """预录视频：按固定顺序循环播，不会听指令改变顺序。"""
    def provider(_frame):
        key = state["key"]
        step = state["step"] % len(VIDEO_SEQUENCE)
        state["step"] += 1
        video_key = VIDEO_SEQUENCE[step]
        if video_key != key:
            return landmarks()
        return landmarks(
            blink=key == "blink",
            mouth=key == "mouth",
            shift=0.02 if key == "motion" else 0.0,
        )
    return provider


def _noise_provider(_frame):
    rng = np.random.default_rng()
    return landmarks(
        blink=bool(rng.random() < 0.1),
        mouth=bool(rng.random() < 0.1),
        shift=float(rng.uniform(-0.05, 0.05)),
    )


SCENARIOS = (
    ("live_person", "真人按提示完成四个动作", _live_provider, False),
    ("static_photo", "举一张静态照片不动", lambda state: _photo_provider, True),
    ("prerecorded_video", "预录视频按固定顺序循环播放", _video_provider, True),
    ("random_motion", "随机乱动，不响应提示", lambda state: _noise_provider, True),
)


def evaluate(rounds: int, timeout: float) -> list[Outcome]:
    detector = fd.register_face_detector()
    real_sleep = fd.time.sleep
    fd.time.sleep = lambda _s: None
    results: list[Outcome] = []
    try:
        for name, description, provider_factory, _is_attack in SCENARIOS:
            passed = 0
            started = time.perf_counter()
            for _ in range(rounds):
                state = {"key": "pose", "calls": 0, "step": 0}
                provider = provider_factory(state)
                if run_liveness(
                    CAMERA,
                    prompt=_prompt_factory(state),
                    timeout=timeout,
                    challenges=detector.liveness_order(),
                    landmark_provider=provider,
                ):
                    passed += 1
            results.append(
                Outcome(name, description, rounds, passed, time.perf_counter() - started)
            )
    finally:
        fd.time.sleep = real_sleep
    return results


def build_report(results: list[Outcome], rounds: int, timeout: float) -> dict:
    scenarios = {}
    for item in results:
        scenarios[item.name] = {
            "description": item.description,
            "rounds": item.rounds,
            "passed": item.passed,
            "pass_rate": round(item.pass_rate, 4),
            "elapsed_seconds": round(item.elapsed, 3),
        }
    photo_block = 1 - scenarios["static_photo"]["pass_rate"]
    video_block = 1 - scenarios["prerecorded_video"]["pass_rate"]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rounds_per_scenario": rounds,
        "challenge_timeout_seconds": timeout,
        "challenge_order": "每轮随机打乱",
        "scenarios": scenarios,
        "summary": {
            "false_reject_rate_live": round(1 - scenarios["live_person"]["pass_rate"], 4),
            "static_photo_block_rate": round(photo_block, 4),
            "prerecorded_video_block_rate": round(video_block, 4),
        },
        "honest_note": (
            "静态照片能被拦住；但预录视频按固定顺序包含全部四种动作时，"
            "当前实现会放行——随机挑战顺序只能提高成本，不能替代真正的"
            "深度/红外或多帧纹理活体。此结论来自实测，不做粉饰。"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=200, help="每个场景的轮数")
    parser.add_argument("--timeout", type=float, default=8.0, help="每个挑战的超时秒数")
    parser.add_argument("--no-write", action="store_true", help="只打印，不写 JSON")
    args = parser.parse_args()

    results = evaluate(args.n, args.timeout)
    report = build_report(results, args.n, args.timeout)

    print(f"\n每场景 {args.n} 轮｜每挑战 {args.timeout}s｜挑战顺序每轮随机")
    print(f"{'场景':<22}{'通过率':>10}{'拦截率':>10}{'耗时':>10}")
    for item in results:
        print(
            f"{item.name:<22}{item.pass_rate:>9.1%}{1 - item.pass_rate:>10.1%}"
            f"{item.elapsed:>9.2f}s"
        )
    print("\n" + report["honest_note"])

    if not args.no_write:
        output = ROOT / "reports" / "liveness.json"
        output.parent.mkdir(exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n报告已写入 {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
