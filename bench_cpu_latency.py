"""CPU 单帧人脸特征提取耗时实测（对应简历「约 0.2s/帧」）。

    python bench_cpu_latency.py                 # 默认用 ultralytics 自带的 zidane.jpg
    python bench_cpu_latency.py face.jpg -n 50

测的就是 extract_face_feature 的端到端：MTCNN（P/R/O 三级级联 + 对齐）
→ InceptionResnetV1 提 512 维特征，全程 CPU。
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

#: 简历那句话说的是 CPU 推理，所以哪怕机器上有 CUDA 也强制走 CPU
os.environ["SECURITY_FORCE_CPU"] = "1"

import cv2  # noqa: E402
import torch  # noqa: E402

from mods.face_detector import device, extract_face_feature  # noqa: E402


def default_image() -> Path:
    import ultralytics
    return Path(ultralytics.__file__).parent / "assets" / "zidane.jpg"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default="")
    ap.add_argument("-n", type=int, default=30)
    #: 摄像头现在就是 640×480；简历说的「约 0.2s/帧」也按这个口径测。
    ap.add_argument("--size", default="640x480", help="先缩到这个尺寸再测，0 表示用原图")
    ap.add_argument("--target", type=float, default=0.2)
    args = ap.parse_args()

    path = Path(args.image) if args.image else default_image()
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"读不到图片: {path}")
        return 2
    if args.size != "0":
        w, h = (int(x) for x in args.size.lower().split("x"))
        frame = cv2.resize(frame, (w, h))

    if extract_face_feature(frame) is None:
        print(f"MTCNN 在 {path.name} 里没找到人脸，换一张正脸图再测")
        return 2

    for _ in range(3):
        extract_face_feature(frame)

    times = []
    for _ in range(max(1, args.n)):
        t0 = time.perf_counter()
        feat = extract_face_feature(frame)
        times.append(time.perf_counter() - t0)
        if feat is None:
            print("中途丢了人脸，这次测量作废")
            return 2

    mean = statistics.mean(times)
    p50 = statistics.median(times)
    p95 = sorted(times)[max(0, int(len(times) * 0.95) - 1)]
    low = min(times)

    print(f"设备: {device}｜torch 线程: {torch.get_num_threads()}｜"
          f"特征维度: {feat.size}")
    print(f"样本: {path.name}｜{frame.shape[1]}x{frame.shape[0]}｜{len(times)} 次")
    print(f"最快 {low * 1000:.0f} ms｜中位 {p50 * 1000:.0f} ms｜"
          f"均值 {mean * 1000:.0f} ms｜P95 {p95 * 1000:.0f} ms")
    ok = p50 <= args.target
    print(f"验收线 {args.target * 1000:.0f} ms（简历口径「约 0.2s/帧」）："
          f"{'达标' if ok else '不达标'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
