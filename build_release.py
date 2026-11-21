"""把项目打成 Windows 免安装目录，并压缩成可上传 GitHub Release 的 zip。

用法：
    pip install pyinstaller
    python build_release.py

产物：
    dist/FaceSecurityTerminal/           # 双击 FaceSecurityTerminal.exe 即可运行
    dist/FaceSecurityTerminal-windows-x64.zip

说明：
  - 打包版默认使用 SQLite（%APPDATA%\\SecuritySystem\\security.db），
    下载者不需要安装 Python / MySQL；
  - 模型权重与 mediapipe 资源都会随包复制，首次运行不再联网下载；
  - 本机若是 CUDA 版 torch，包里会带上 CUDA 运行库（体积增大数 GB），
    只想做通用分发时建议先装 CPU 版：
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
BUILD = ROOT / "build"
APP_NAME = "FaceSecurityTerminal"

REQUIRED_MODELS = [
    ROOT / "face_landmarker.task",
    ROOT / "yolov8n.pt",
    ROOT / "facenet_pytorch_data" / "pnet.pt",
    ROOT / "facenet_pytorch_data" / "rnet.pt",
    ROOT / "facenet_pytorch_data" / "onet.pt",
]


def check_assets() -> None:
    missing = [str(p.relative_to(ROOT)) for p in REQUIRED_MODELS if not p.exists()]
    if missing:
        raise SystemExit("缺少模型文件，无法打包：" + "、".join(missing))


def warn_if_cuda_torch() -> None:
    try:
        import torch
    except ImportError:
        return
    if getattr(torch.version, "cuda", None):
        print(
            f"[警告] 当前 torch 带 CUDA {torch.version.cuda}，打包体积会非常大。\n"
            "        做通用分发建议改用 CPU 版：\n"
            "        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
        )


def build_command() -> list[str]:
    # --console 便于排查首次运行问题（能直接看到 print 的报错）
    window_mode = "--console" if "--console" in sys.argv else "--windowed"
    datas = [
        (ROOT / "face_landmarker.task", "."),
        (ROOT / "yolov8n.pt", "."),
        (ROOT / "logo", "logo"),
    ]
    for weight in sorted((ROOT / "facenet_pytorch_data").glob("*.pt")):
        datas.append((weight, "facenet_pytorch_data"))

    command = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean", window_mode,
        "--name", APP_NAME,
        "--distpath", str(DIST),
        "--workpath", str(BUILD),
        "--specpath", str(BUILD),
        "--collect-all", "mediapipe",
        "--collect-all", "facenet_pytorch",
        "--collect-all", "ultralytics",
        "--hidden-import", "pymysql",
        "--exclude-module", "matplotlib.tests",
        "--exclude-module", "tkinter",
    ]
    for source, target in datas:
        command += ["--add-data", os.pathsep.join([str(source), target])]
    command.append(str(ROOT / "main.py"))
    return command


def zip_output() -> Path:
    app_dir = DIST / APP_NAME
    archive = shutil.make_archive(
        str(DIST / f"{APP_NAME}-windows-x64"), "zip",
        root_dir=str(DIST), base_dir=APP_NAME,
    )
    print(f"[完成] 目录: {app_dir}")
    print(f"[完成] 压缩包: {archive}（约 {Path(archive).stat().st_size / 1024 / 1024:.0f} MB）")
    print("下一步：GitHub → Releases → Draft a new release → 上传这个 zip。")
    return Path(archive)


def main() -> None:
    check_assets()
    warn_if_cuda_torch()
    print("[1/2] PyInstaller 打包中（首次较慢）...")
    subprocess.run(build_command(), cwd=str(ROOT), check=True)
    print("[2/2] 压缩...")
    zip_output()


if __name__ == "__main__":
    main()
