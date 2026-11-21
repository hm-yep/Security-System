"""项目一简历条款验收：全程不接摄像头、不真锁屏、不真拦截。

    python verify_claims.py
"""

from __future__ import annotations

import inspect
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("  OK  " if cond else "  FAIL ") + name + (f"｜{detail}" if detail else ""))
    if not cond:
        fails.append(name)


class P:
    __slots__ = ("x", "y")

    def __init__(self, x=0.5, y=0.5):
        self.x, self.y = x, y


def landmarks(*, blink=False, mouth=False, shift=0.0):
    """造一份 478 点的脸：默认睁眼、闭嘴、正脸；按开关改成眨眼/张嘴/位移。"""
    pts = [P() for _ in range(478)]
    for base, idx in ((0.4, [33, 160, 158, 133, 153, 144]),
                      (0.6, [362, 387, 385, 263, 373, 380])):
        a, b, c, d, e, f = idx
        pts[a] = P(base, 0.5)
        pts[d] = P(base + 0.1, 0.5)
        if blink:
            pts[b] = pts[c] = pts[e] = pts[f] = P(base + 0.05, 0.5)
        else:
            pts[b] = P(base + 0.02, 0.52)
            pts[f] = P(base + 0.02, 0.48)
            pts[c] = P(base + 0.08, 0.52)
            pts[e] = P(base + 0.08, 0.48)
    m1, m2, m3, m4, m5, m6 = 61, 78, 13, 14, 308, 291
    pts[m1] = P(0.45, 0.6)
    pts[m4] = P(0.55, 0.6)
    if mouth:
        pts[m2] = P(0.47, 0.65)
        pts[m6] = P(0.47, 0.55)
        pts[m3] = P(0.53, 0.65)
        pts[m5] = P(0.53, 0.55)
    else:
        pts[m2] = P(0.47, 0.601)
        pts[m6] = P(0.47, 0.599)
        pts[m3] = P(0.53, 0.601)
        pts[m5] = P(0.53, 0.599)
    pts[1] = P(0.5, 0.5)
    pts[33] = P(0.4, 0.5)
    pts[133] = P(0.6, 0.5)
    if shift:
        for i in (1, 33, 133, 61, 291):
            pts[i] = P(pts[i].x + shift, pts[i].y)
    return pts


class FakeDetector:
    """按弹窗文案判断当前挑战，返回对应该挑战的关键点。"""

    def __init__(self, fail: set[str] | None = None):
        self.ifcamera = True
        self.fail = fail or set()
        self.mode = "pose"
        self.calls = 0
        self.prompts: list[str] = []
        self.released = False
        self.camera = self

    def show_auto_close_message(self, text, duration):
        self.prompts.append(text)
        self.calls = 0
        if "眨眼" in text:
            self.mode = "blink"
        elif "张嘴" in text:
            self.mode = "mouth"
        elif "正对" in text:
            self.mode = "pose"
        elif "稍等" in text:
            self.mode = "motion"

    def get_landmarks(self, timeout=None):
        self.calls += 1
        if self.mode in self.fail:
            return None
        if self.mode == "blink":
            return landmarks(blink=True)
        if self.mode == "mouth":
            return landmarks(mouth=True)
        if self.mode == "motion":
            return landmarks(shift=0.0 if self.calls == 1 else 0.02)
        return landmarks()

    def get_frame(self):
        return None

    def release(self):
        self.released = True


class NoFrameCam:
    def __init__(self):
        self.released = False

    def getframe(self):
        return None

    def release(self):
        self.released = True


def liveness_checks():
    print("\n[1] 双重活体：四步随机状态机")
    from mods import face_detector as fd
    from mods.face_detector import register_face_detector

    det = register_face_detector()
    keys = {"blink", "mouth", "pose", "motion"}

    orders = set()
    for _ in range(40):
        orders.add(tuple(c.key for c in det.liveness_order()))
    check("每轮恰好四个挑战", all(set(o) == keys and len(o) == 4 for o in orders))
    check("顺序确实被打乱", len(orders) > 1, f"{len(orders)} 种顺序 / 40 轮")

    real_sleep = fd.time.sleep
    fd.time.sleep = lambda _s: None
    try:
        seen_orders = set()
        all_ok = True
        for _ in range(12):
            d = FakeDetector()
            det.hand_in = lambda is_auto: np.zeros(512, dtype=np.float32)
            ok, feat = det.ifpass(detector=d, timeout=1.0)
            seen_orders.add(tuple(x for x in d.prompts if "超时" not in x))
            all_ok = all_ok and ok and feat is not None and feat.size == 512
        check("四个挑战全过才返回特征", all_ok)
        check("ifpass 每轮换顺序", len(seen_orders) > 1, f"{len(seen_orders)} 种")

        d = FakeDetector(fail={"mouth"})
        det.hand_in = lambda is_auto: np.zeros(512, dtype=np.float32)
        ok, feat = det.ifpass(detector=d, timeout=0.6)
        check("漏做一个挑战就失败", ok is False and feat is None)
        check("失败会提示具体项目", any("张嘴检测超时" in p for p in d.prompts))
        check("失败也会释放摄像头", d.released)
    finally:
        fd.time.sleep = real_sleep

    det.prev_landmarks = None
    first = det.check_motion(landmarks())
    second = det.check_motion(landmarks(shift=0.02))
    check("微动第一帧只当基线、不算通过", first is False and second is True)

    u = fd.userfaceinfo.__new__(fd.userfaceinfo)
    u.ifcamera = True
    u.camera = NoFrameCam()
    t0 = time.time()
    got = u.get_landmarks(timeout=0.2)
    dt = time.time() - t0
    check("没脸时按超时返回，不会死等", got is None and 0.15 <= dt < 1.5, f"{dt:.2f}s")

    check("判据仍是 EAR / MAR / 头部姿态 / 微动",
          all(hasattr(det, n) for n in ("check_blink", "check_mouth_open",
                                        "check_head_pose", "check_motion")))


def lock_liveness_checks():
    print("\n[1b] 锁屏解锁必须过活体")
    from mods import face_detector as fd
    from mods.face_detector import LivenessChallenge, run_liveness

    class FrameCam:
        def getframe(self):
            return "frame"

    state = {"key": "pose", "calls": 0}

    def prompt(text):
        for ch in fd.LIVENESS_CHALLENGES:
            if ch.prompt == text:
                state["key"] = ch.key
                state["calls"] = 0

    def lifelike(_frame):
        state["calls"] += 1
        key = state["key"]
        #: 微动要两帧不同：第一帧基线、第二帧位移 0.02
        shift = 0.02 if key == "motion" and state["calls"] > 1 else 0.0
        return landmarks(blink=key == "blink", mouth=key == "mouth", shift=shift)

    order = [LivenessChallenge(c.key, c.prompt, 0.0, c.timeout_msg)
             for c in fd.LIVENESS_CHALLENGES]
    ok = run_liveness(FrameCam(), prompt=prompt, timeout=1.0,
                      challenges=order, landmark_provider=lifelike)
    check("真人按提示做完四步才通过", ok is True)

    ok = run_liveness(FrameCam(), prompt=lambda _t: None, timeout=0.2,
                      challenges=order[:1],
                      landmark_provider=lambda _f: landmarks())
    check("举张照片过不了眨眼", ok is False)


def system_checks():
    print("\n[2] 系统层防护")
    from mods.auto_ui import FaceDetectionThread, InputGuard, Lockscreen
    from mods.hand_ui import lock_workstation_api

    check("单 Win 键被拦", InputGuard.should_block(0x5B))
    check("Win+D 被拦", InputGuard.should_block(0x44, win=True))
    check("Alt+Tab 被拦", InputGuard.should_block(0x09, alt=True))
    check("Alt+F4 被拦", InputGuard.should_block(0x73, alt=True))
    check("Ctrl+Esc 被拦", InputGuard.should_block(0x1B, ctrl=True))
    check("普通字母键放行", not InputGuard.should_block(0x41))
    check("Ctrl+C 放行", not InputGuard.should_block(0x43, ctrl=True))

    check("鼠标按键/滚轮/侧键全拦",
          all(InputGuard.should_block_mouse(m)
              for m in (0x0201, 0x0204, 0x0207, 0x020A, 0x020B, 0x020E)))
    check("鼠标移动也拦", InputGuard.should_block_mouse(0x0200))
    check("未知消息不误伤", not InputGuard.should_block_mouse(0x00FF))

    guard = InputGuard()
    installed = guard.start()
    check("低级键鼠钩子能真实装上",
          installed and guard._keyboard_hook and guard._mouse_hook,
          f"keyboard={bool(guard._keyboard_hook)} mouse={bool(guard._mouse_hook)}")
    guard.stop()
    check("摘钩子后两个句柄都清空", not guard._keyboard_hook and not guard._mouse_hook)

    check("锁屏窗口吞掉按键事件", "accept()" in inspect.getsource(Lockscreen.keyPressEvent))
    check("锁屏期间拒绝关窗", "ignore()" in inspect.getsource(Lockscreen.closeEvent))
    check("解锁会摘掉键鼠钩子", hasattr(Lockscreen, "stop_input_guard"))

    run_src = inspect.getsource(FaceDetectionThread.run)
    check("锁屏匹配身份后必须过活体", "run_liveness" in run_src)
    check("活体在发解锁信号之前", run_src.index("run_liveness") < run_src.index("result_signal.emit"))

    check("LockWorkStation 成功路径", lock_workstation_api(lambda: 1) is True)
    check("LockWorkStation 失败会如实返回", lock_workstation_api(lambda: 0) is False)

    src = (ROOT / "mods" / "hand_ui.py").read_text(encoding="utf-8")
    check("托盘有系统锁屏入口", "系统锁屏（Windows API）" in src)
    check("自定义锁屏失败会退回系统锁屏", "self.lock_workstation()" in src)
    init_src = inspect.getsource(Lockscreen.__init__)
    check("无边框置顶窗口", "Qt.FramelessWindowHint" in init_src
          and "Qt.WindowStaysOnTopHint" in init_src)


def yolo_checks():
    print("\n[3] YOLOv8 多目标人体检测")
    import cv2
    import ultralytics
    from ultralytics import YOLO

    asset = Path(ultralytics.__file__).parent / "assets" / "zidane.jpg"
    model_path = ROOT / "yolov8n.pt"
    if not asset.exists() or not model_path.exists():
        check("找到测试图与模型", False, f"asset={asset.exists()} model={model_path.exists()}")
        return
    img = cv2.imread(str(asset))
    results = YOLO(str(model_path))(img, verbose=False)
    persons = []
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                persons.append(((x2 - x1) * (y2 - y1), x1, y1, x2, y2))
    persons.sort(reverse=True)
    check("一帧内检出多个 person", len(persons) >= 2, f"{len(persons)} 个")
    check("按面积挑最大目标做人脸比对", persons == sorted(persons, reverse=True))


def audit_checks():
    print("\n[4] 审计 / 自启 / 托盘（静态核对）")
    src = (ROOT / "mods" / "hand_ui.py").read_text(encoding="utf-8")
    check("登录写审计", "登录系统" in src and "insertdatalog" in src)
    check("注册写审计", "注册成功" in src)
    check("改密写审计", "修改了登录密码" in src)
    check("有日志查询界面", "操作日志" in src and "获取操作日志" in src)
    startup = (ROOT / "mods" / "startup_manager.py").read_text(encoding="utf-8")
    check("注册表开机自启", "HKEY_CURRENT_USER" in startup and "CurrentVersion\\Run" in startup)
    check("系统托盘常驻", "QSystemTrayIcon" in src)


def main() -> int:
    liveness_checks()
    lock_liveness_checks()
    system_checks()
    yolo_checks()
    audit_checks()
    print("\n结论：", "全部通过" if not fails else f"{len(fails)} 项没过 -> {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())

