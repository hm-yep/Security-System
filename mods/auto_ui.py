# mods/auto_ui.py
from ultralytics import YOLO
import cv2
import sys
import os
import numpy as np
import torch
import ctypes
from ctypes import wintypes
import time
from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QMessageBox, QApplication
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from .database import package
from .feature_codec import parse_feature as decode_feature
from .matching import match_feature, face_threshold
from .face_detector import auto_hand_controller, mtcnn, resnet, device, run_liveness, get_base_dir
from .camera import camera_process

DEBUG_MATCH = False

LOCK_STYLE = """
QWidget {
    background-color: #0a0a1a;
}
QLabel#title {
    color: #4a9eff;
    font-size: 42px;
    font-weight: 300;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
}
QLabel#status {
    color: #aab8d0;
    font-size: 20px;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    background-color: rgba(0, 0, 0, 0.5);
    border-radius: 25px;
    padding: 12px 30px;
}
QLabel#video {
    border: 3px solid rgba(74, 158, 255, 0.25);
    border-radius: 20px;
    background-color: #000000;
}
"""

LRESULT = ctypes.c_ssize_t
_HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)


class InputGuard:
    """锁屏期间的键鼠事件屏蔽（防关闭）。

    Qt 里的 event.accept() 只能吞掉发给本窗口的事件；鼠标可以点去副屏或别的
    窗口把焦点送出去，Win 键、Alt+Tab、Alt+F4 这类系统快捷键根本不经过窗口，
    所以再加一层 WH_KEYBOARD_LL / WH_MOUSE_LL 低级钩子：
      - 键盘只拦"能绕过锁屏"的组合键，普通按键照常放行；
      - 鼠标在锁屏期间全量吞掉（移动、按键、滚轮、侧键）——锁屏里没有任何
        可点的控件，放行只会把焦点交给别的窗口。
    Ctrl+Alt+Del 由系统安全桌面接管，用户态钩子拦不住，这也是永远留着的出口。
    """

    WH_KEYBOARD_LL = 13
    WH_MOUSE_LL = 14
    WM_KEYDOWN = 0x0100
    WM_SYSKEYDOWN = 0x0104
    #: WH_MOUSE_LL 会送来的全部消息：移动、左/右/中键按下抬起、滚轮、
    #: 侧键按下抬起、横向滚轮。锁屏里一个都不放。
    BLOCKED_MOUSE = frozenset({
        0x0200, 0x0201, 0x0202, 0x0204, 0x0205,
        0x0207, 0x0208, 0x020A, 0x020B, 0x020C, 0x020E,
    })
    VK_TAB, VK_ESCAPE, VK_F4 = 0x09, 0x1B, 0x73
    VK_LWIN, VK_RWIN = 0x5B, 0x5C
    VK_D, VK_E, VK_R, VK_L = 0x44, 0x45, 0x52, 0x4C
    VK_MENU, VK_CONTROL = 0x12, 0x11

    class KBDLLHOOKSTRUCT(ctypes.Structure):
        _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD),
                    ("flags", wintypes.DWORD), ("time", wintypes.DWORD),
                    ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))]

    @classmethod
    def should_block(cls, vk, alt=False, ctrl=False, win=False):
        """哪些按键必须吞掉：单 Win 键、Win+D/E/R/L、Alt+Tab/Esc/F4、Ctrl+Esc。"""
        if vk in (cls.VK_LWIN, cls.VK_RWIN):
            return True
        if win and vk in (cls.VK_D, cls.VK_E, cls.VK_R, cls.VK_L):
            return True
        if alt and vk in (cls.VK_TAB, cls.VK_ESCAPE, cls.VK_F4):
            return True
        if ctrl and vk == cls.VK_ESCAPE:
            return True
        return False

    @classmethod
    def should_block_mouse(cls, msg):
        """锁屏期间鼠标消息一律吞掉：没有可交互控件，放行只会把焦点送出去。"""
        return msg in cls.BLOCKED_MOUSE

    def __init__(self):
        self._user32 = None
        self._keyboard_hook = None
        self._mouse_hook = None
        self._key_cb = None
        self._mouse_cb = None

    def start(self):
        if self._keyboard_hook or os.name != "nt":
            return False
        u = ctypes.WinDLL("user32", use_last_error=True)
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        u.SetWindowsHookExW.argtypes = [ctypes.c_int, _HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD]
        u.SetWindowsHookExW.restype = wintypes.HHOOK
        u.CallNextHookEx.argtypes = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
        u.CallNextHookEx.restype = LRESULT
        u.UnhookWindowsHookEx.argtypes = [wintypes.HHOOK]
        u.UnhookWindowsHookEx.restype = wintypes.BOOL
        u.GetAsyncKeyState.argtypes = [ctypes.c_int]
        u.GetAsyncKeyState.restype = ctypes.c_short
        #: 不声明 restype 时 HMODULE 会被默认截成 32 位，SetWindowsHookExW 报 126（模块不存在）
        k.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        k.GetModuleHandleW.restype = wintypes.HMODULE
        self._user32 = u
        #: 回调对象必须存住，被回收就等于钩子指向野指针
        self._key_cb = _HOOKPROC(self._on_key)
        self._mouse_cb = _HOOKPROC(self._on_mouse)
        module = k.GetModuleHandleW(None)
        self._keyboard_hook = u.SetWindowsHookExW(self.WH_KEYBOARD_LL, self._key_cb, module, 0)
        key_err = ctypes.get_last_error()
        self._mouse_hook = u.SetWindowsHookExW(self.WH_MOUSE_LL, self._mouse_cb, module, 0)
        mouse_err = ctypes.get_last_error()
        if not self._keyboard_hook:
            print(f"[锁屏] 键盘钩子安装失败: {key_err}")
        if not self._mouse_hook:
            print(f"[锁屏] 鼠标钩子安装失败: {mouse_err}")
        if not (self._keyboard_hook and self._mouse_hook):
            self.stop()  # 只装上一半等于没装：把另一个也摘掉，避免留下半套钩子
            return False
        return True

    def stop(self):
        if self._user32 is None:
            return
        for hook in (self._mouse_hook, self._keyboard_hook):
            if hook:
                self._user32.UnhookWindowsHookEx(hook)
        self._keyboard_hook = None
        self._mouse_hook = None

    def _on_key(self, n_code, w_param, l_param):
        if n_code >= 0 and w_param in (self.WM_KEYDOWN, self.WM_SYSKEYDOWN):
            kb = ctypes.cast(l_param, ctypes.POINTER(self.KBDLLHOOKSTRUCT)).contents
            u = self._user32
            alt = bool(u.GetAsyncKeyState(self.VK_MENU) & 0x8000)
            ctrl = bool(u.GetAsyncKeyState(self.VK_CONTROL) & 0x8000)
            win = bool((u.GetAsyncKeyState(self.VK_LWIN) | u.GetAsyncKeyState(self.VK_RWIN)) & 0x8000)
            if self.should_block(kb.vkCode, alt=alt, ctrl=ctrl, win=win):
                return 1
        return self._user32.CallNextHookEx(None, n_code, w_param, l_param)

    def _on_mouse(self, n_code, w_param, l_param):
        if n_code >= 0 and self.should_block_mouse(w_param):
            return 1
        return self._user32.CallNextHookEx(None, n_code, w_param, l_param)


class Hodorunlock():
    def __init__(self):
        self.pipe_name = r"\\.\pipe\CredentialProviderPipe"

    def unlock(self, username, password, timeout_ms=3000):
        try:
            handle = ctypes.windll.kernel32.CreateFileW(
                self.pipe_name, 0xC0000000, 0, None, 3, 0, None
            )
            if handle == -1:
                return False
            cmd = f"UNLOCK:.\\{username}:{password}\n"
            cmd_bytes = cmd.encode('utf-8')
            bytes_written = ctypes.c_ulong(0)
            success = ctypes.windll.kernel32.WriteFile(
                handle, cmd_bytes, len(cmd_bytes), ctypes.byref(bytes_written), None
            )
            ctypes.windll.kernel32.CloseHandle(handle)
            return bool(success)
        except Exception as e:
            print(f"[Hodor] 异常: {e}")
            return False


class FaceDetectionThread(QThread):
    result_signal = pyqtSignal(str, bool)
    status_signal = pyqtSignal(str, str)

    def __init__(self, camera, detector, current_user=None):
        super().__init__()
        self.camera = camera
        self.detector = detector
        self.current_user = current_user
        self.running = True
        self.interval = 3.0
        self.retry_count = 0
        self.liveness_timeout = 10.0
        #: 失败状态每轮都会重复出现，审计按时间节流，避免把日志表刷爆
        self._last_liveness_log = 0.0
        self._last_mismatch_log = 0.0

    def _audit(self, message):
        try:
            package().insertdatalog(message, self.current_user or "System")
        except Exception as e:
            print(f"[检测线程] 审计写入失败: {e}")

    def run(self):
        pkg = package()
        while self.running:
            try:
                try:
                    if pkg.is_empty():
                        self.status_signal.emit("⚠️ 暂无注册用户，请先手动注册", "color: #ff6b6b;")
                        time.sleep(self.interval)
                        continue
                except AttributeError:
                    pass

                success, userface = self.detector.run(True, camera=self.camera)

                if not success:
                    self.status_signal.emit("❌ 未检测到人脸，请正对摄像头", "color: #ff6b6b;")
                    self.retry_count += 1
                    if self.retry_count > 3:
                        self.status_signal.emit("⚠️ 检测器异常，重新初始化...", "color: #ffaa00;")
                        self.detector = auto_hand_controller()
                        self.retry_count = 0
                    time.sleep(self.interval)
                    continue

                self.retry_count = 0
                matched_user = self.match_current_user(pkg, userface)
                if not matched_user:
                    now = time.time()
                    if now - self._last_mismatch_log > 60:
                        self._last_mismatch_log = now
                        self._audit(f"人脸不匹配：{self.current_user} 验证失败")
                    self.status_signal.emit("❌ 人脸不匹配，请重试", "color: #ff6b6b;")
                    time.sleep(self.interval)
                    continue

                #: 身份匹配只解决"是不是这个人"，照片也能匹配上；
                #: 解锁前必须再跑一次分步挑战，确认镜头前是活人。
                self.status_signal.emit(f"✅ 身份匹配：{matched_user}，请完成活体检测", "color: #4a9eff;")
                if not run_liveness(self.camera, prompt=self._liveness_prompt,
                                    timeout=self.liveness_timeout):
                    now = time.time()
                    if now - self._last_liveness_log > 30:
                        self._last_liveness_log = now
                        self._audit(f"活体拦截：{self.current_user} 未完成动作挑战")
                    self.status_signal.emit("❌ 活体检测未通过：照片/视频做不出指定动作",
                                            "color: #ff6b6b;")
                    self.retry_count += 1
                    time.sleep(self.interval)
                    continue

                self._audit(f"人脸解锁成功：{matched_user}")
                self.status_signal.emit(f"✅ 活体通过：{matched_user}，正在解锁...", "color: #51cf66;")
                self.result_signal.emit(matched_user, True)
                break

            except Exception as e:
                print(f"[检测线程] 异常: {e}")
                self.status_signal.emit(f"⚠️ 检测异常，正在重置...", "color: #ffaa00;")
                self.detector = auto_hand_controller()
                self.retry_count = 0
                time.sleep(2.0)

    def _liveness_prompt(self, text):
        color = "#ff6b6b;" if "❌" in text else "#ffd43b;"
        self.status_signal.emit(text, f"color: {color}")

    def parse_feature(self, feat_str):
        """委托给统一编解码器：只解析数据，绝不执行数据库里的内容。

        旧实现最后一步是 eval(feat_str)，而 feat_str 来自 userinfo 表，
        相当于把数据库字段当 Python 代码执行，已从所有路径移除。
        """
        return decode_feature(feat_str)

    def match_current_user(self, pkg, feature):
        if self.current_user is None:
            return None
        stored_text = pkg.get_face_feature(self.current_user)
        if not stored_text:
            return None

        sim, passed = match_feature(stored_text, feature)
        if sim is None:
            print(f"[匹配调试] 当前用户 {self.current_user} 的特征格式或维度异常，拒绝解锁")
            return None
        print(f"[匹配调试] 当前用户 {self.current_user}, 相似度: {sim:.4f}, 阈值: {face_threshold()}")
        return self.current_user if passed else None

    def stop(self):
        self.running = False
        self.wait()


class Lockscreen(QWidget):
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Window
        )
        self.setStyleSheet(LOCK_STYLE)
        self.showFullScreen()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus(Qt.ActiveWindowFocusReason)

        self.camera = None
        self.is_unlocked = False
        self.detector = auto_hand_controller()
        self.timer = None
        self._matched_user = None
        self.main_window = main_window
        self.current_user = self.main_window.username if self.main_window else None
        self.detection_thread = None

        self.setup_ui()
        self.start_camera()
        self.start_detection_thread()

        self.yolov8 = None
        self.yolotimer = None
        try:
            # 用绝对路径：打包后工作目录不一定是 exe 所在目录
            self.yolov8 = YOLO(os.path.join(get_base_dir(), 'yolov8n.pt'))
            self.stranger_count = 0
            self.stranger_threshold = 5
            self.yolotimer = QTimer(self)
            self.yolotimer.timeout.connect(self.check_stranger)
            self.yolotimer.start(2000)
        except Exception as e:
            print(f"[陌生人检测] YOLOv8 不可用: {e}")

        #: 键鼠屏蔽最后装：窗口已经全屏置顶，钩子才开始拦系统快捷键和全部鼠标
        self.input_guard = InputGuard()
        if not self.input_guard.start():
            print("[锁屏] 低级键鼠钩子没装上，只剩 Qt 层事件拦截")
        self.grabKeyboard()
        self.grabMouse()

    def setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 50, 0, 50)
        main_layout.setSpacing(20)
        main_layout.setAlignment(Qt.AlignCenter)

        self.title_label = QLabel("🔐 Security System")
        self.title_label.setObjectName("title")
        self.title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.title_label)

        video_container = QHBoxLayout()
        video_container.setAlignment(Qt.AlignCenter)

        self.video_label = QLabel()
        self.video_label.setObjectName("video")
        self.video_label.setFixedSize(480, 360)
        self.video_label.setScaledContents(True)
        video_container.addWidget(self.video_label)
        main_layout.addLayout(video_container)

        self.status_label = QLabel("正在启动摄像头...")
        self.status_label.setObjectName("status")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)

        self.setLayout(main_layout)

        black = QImage(480, 360, QImage.Format_RGB888)
        black.fill(Qt.black)
        self.video_label.setPixmap(QPixmap.fromImage(black))

    def start_camera(self):
        self.camera = camera_process()
        if not self.camera.start():
            QMessageBox.critical(self, "错误", "无法打开摄像头！")
            self.close()
            QApplication.quit()
            return

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)

    def update_frame(self):
        if self.camera is None:
            return
        frame = self.camera.getframe()
        if frame is not None:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    def start_detection_thread(self):
        if self.detection_thread is not None:
            self.detection_thread.stop()
        self.detection_thread = FaceDetectionThread(self.camera, self.detector, self.current_user)
        self.detection_thread.status_signal.connect(self.update_status)
        self.detection_thread.result_signal.connect(self.on_detection_result)
        self.detection_thread.start()

    def stop_detection_thread(self):
        if self.detection_thread is not None:
            self.detection_thread.stop()
            self.detection_thread = None

    def update_status(self, message, style):
        self.status_label.setText(message)
        self.status_label.setStyleSheet(style)

    def on_detection_result(self, username, success):
        if success and not self.is_unlocked:
            self._matched_user = username
            self.update_status(f"✅ 人脸识别 + 活体检测通过：{username}，正在解锁...", "color: #51cf66;")
            QTimer.singleShot(500, self.unlock_system)

    def unlock_system(self):
        if self.is_unlocked:
            return
        self.is_unlocked = True

        self.stop_detection_thread()
        if self.timer:
            self.timer.stop()
        if self.yolotimer:
            self.yolotimer.stop()
        self.stop_input_guard()
        if self.camera:
            self.camera.release()
        self.close()

    def stop_input_guard(self):
        """解锁/关窗时必须先摘钩子，再放键鼠——顺序反了会留一个卡住系统的空档。"""
        guard = getattr(self, "input_guard", None)
        if guard is not None:
            guard.stop()
            self.input_guard = None
        self.releaseKeyboard()
        self.releaseMouse()

        if self.main_window is not None:
            self.main_window.show()
            self.main_window.raise_()
            if hasattr(self.main_window, 'lock_screen_active'):
                self.main_window.lock_screen_active = False
                self.main_window.lock_screen_window = None
        else:
            from .hand_ui import mainwindow
            import mods.hand_ui
            if self._matched_user:
                mods.hand_ui.UN = self._matched_user
            self.main_win = mainwindow()
            self.main_win.show()

    #: 下面四个 accept 而不是 ignore：ignore 只是"我不处理"，事件会继续往上传；
    #: 锁屏要的是"到此为止"，键鼠一律吞掉，配合 grabKeyboard/grabMouse 形成闭环。
    def keyPressEvent(self, event):
        event.accept()

    def mousePressEvent(self, event):
        event.accept()

    def mouseMoveEvent(self, event):
        event.accept()

    def mouseReleaseEvent(self, event):
        event.accept()

    def contextMenuEvent(self, event):
        event.accept()

    def closeEvent(self, event):
        if not self.is_unlocked:
            #: Alt+F4 / 点关闭 / 系统发来的关窗请求，统统拒绝
            event.ignore()
        else:
            self.stop_detection_thread()
            self.stop_input_guard()
            if self.camera:
                self.camera.release()
            event.accept()

    def check_stranger(self):
        if self.is_unlocked or self.yolov8 is None:
            return
        frame = self.camera.getframe()
        if frame is None:
            return
        results = self.yolov8(frame, imgsz=320, verbose=False)
        persons = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    persons.append((area, x1, y1, x2, y2))
        if not persons:
            self.stranger_count = 0
            return
        persons.sort(key=lambda p: p[0], reverse=True)
        _, x1, y1, x2, y2 = persons[0]
        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0:
            return
        rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        face_tensor = mtcnn(rgb)
        if face_tensor is None:
            return
        face_tensor = face_tensor.to(device)
        with torch.no_grad():
            feature = resnet(face_tensor.unsqueeze(0))[0].cpu().numpy()
        pkg = package()
        if self.match_current_user_static(pkg, feature):
            self.stranger_count = 0
        else:
            self.stranger_count += 1
            if self.stranger_count >= self.stranger_threshold:
                self.trigger_stranger_alert()
                self.stranger_count = 0

    def match_current_user_static(self, pkg, feature):
        if self.current_user is None:
            return None
        stored_text = pkg.get_face_feature(self.current_user)
        if not stored_text:
            return None

        sim, passed = match_feature(stored_text, feature)
        print(f"[静态匹配] 当前用户 {self.current_user}, 相似度: {sim}")
        return passed

    def parse_feature(self, feat_str):
        """委托给统一编解码器：只解析数据，绝不执行数据库里的内容。

        旧实现最后一步是 eval(feat_str)，而 feat_str 来自 userinfo 表，
        相当于把数据库字段当 Python 代码执行，已从所有路径移除。
        """
        return decode_feature(feat_str)

    def trigger_stranger_alert(self):
        pkg = package()
        pkg.insertdatalog("检测到陌生人停留", "System")
        print("[陌生人告警] 检测到陌生人长时间停留，已记录日志")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Lockscreen()
    window.show()
    sys.exit(app.exec_())