import cv2
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QTimer
from .camera import camera_process
import time
import random
from dataclasses import dataclass
from facenet_pytorch import MTCNN, InceptionResnetV1
from .face_quality import assess_face, fuse_features
import math
import numpy as np
import os
import sys

# ===== 全局时间戳计数器 =====
_timestamp_counter = 0

# ===== 获取基础目录 =====
def get_base_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = get_base_dir()
print(f"[FaceDetector] 基础目录: {BASE_DIR}")

# ===== MediaPipe 模型路径 =====
landmarker_path = os.path.join(BASE_DIR, 'face_landmarker.task')
if not os.path.exists(landmarker_path):
    alt_path = os.path.join(os.getcwd(), 'face_landmarker.task')
    if os.path.exists(alt_path):
        landmarker_path = alt_path
    else:
        raise FileNotFoundError(f"无法找到 face_landmarker.task，路径: {landmarker_path}")

# ===== MTCNN 模型路径 =====
def get_facenet_model_path():
    base = get_base_dir()
    candidates = [
        os.path.join(base, 'facenet_pytorch', 'data'),
        os.path.join(base, 'facenet_pytorch_data'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'facenet_pytorch_data'),
        os.path.join(os.getcwd(), 'facenet_pytorch_data'),
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.exists(os.path.join(p, 'pnet.pt')):
            print(f"[FaceDetector] 找到模型路径: {p}")
            return p
    try:
        import facenet_pytorch
        pkg_path = os.path.dirname(facenet_pytorch.__file__)
        p = os.path.join(pkg_path, 'data')
        if os.path.exists(p) and os.path.exists(os.path.join(p, 'pnet.pt')):
            print(f"[FaceDetector] 使用 site-packages 模型: {p}")
            return p
    except:
        pass
    print("[FaceDetector] 警告: 未找到模型文件!")
    return candidates[1]

FACENET_DATA_DIR = get_facenet_model_path()
os.environ['FACENET_PYTORCH_DATA'] = FACENET_DATA_DIR


# ===== 混合架构核心绑定 =====
def performance_core_mask():
    """P+E 混合 CPU 上返回 P 核的逻辑处理器掩码；非 Windows / 非混合架构返回 None。

    Windows 默认会把普通优先级的推理线程调度到 E 核：同一帧实测 E 核比 P 核慢
    4~5 倍，「CPU 单帧约 0.2s」只有在 P 核上才稳定。
    """
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class GroupAffinity(ctypes.Structure):
        _fields_ = [("Mask", ctypes.c_size_t), ("Group", wintypes.WORD),
                    ("Reserved", wintypes.WORD * 3)]

    class ProcessorRelationship(ctypes.Structure):
        _fields_ = [("Flags", wintypes.BYTE), ("EfficiencyClass", wintypes.BYTE),
                    ("Reserved", wintypes.BYTE * 20), ("GroupCount", wintypes.WORD),
                    ("GroupMask", GroupAffinity * 1)]

    class ProcessorInfo(ctypes.Structure):
        _fields_ = [("Relationship", ctypes.c_int), ("Size", wintypes.DWORD),
                    ("Processor", ProcessorRelationship)]

    kernel32 = ctypes.windll.kernel32
    kernel32.GetLogicalProcessorInformationEx.argtypes = [
        ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
    size = wintypes.DWORD(0)
    kernel32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))  # 0 = RelationProcessorCore
    if not size.value:
        return None
    buf = ctypes.create_string_buffer(size.value)
    if not kernel32.GetLogicalProcessorInformationEx(
            0, ctypes.cast(buf, ctypes.c_void_p), ctypes.byref(size)):
        return None
    by_class = {}
    offset = 0
    while offset < size.value:
        rec = ProcessorInfo.from_buffer(buf, offset)
        if rec.Relationship == 0 and rec.Processor.GroupCount == 1:
            g = rec.Processor.GroupMask[0]
            by_class.setdefault(rec.Processor.EfficiencyClass, []).append((g.Group, int(g.Mask)))
        offset += rec.Size
    if len(by_class) < 2:
        return None
    best = max(by_class)  # 能效等级更高 = 性能核
    if any(group != 0 for group, _ in by_class[best]):
        return None
    mask = 0
    for _, m in by_class[best]:
        mask |= m
    return mask or None


def pin_to_performance_cores():
    mask = performance_core_mask()
    if not mask:
        return 0
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), mask):
        return 0
    return mask


P_CORE_MASK = pin_to_performance_cores()
if P_CORE_MASK:
    # 线程数按 8 封顶：MTCNN/ResNet 的层都很小，线程再多只会互相抢核，
    # 实测 24 线程比 8 线程慢约 30%。
    torch.set_num_threads(max(1, min(8, bin(P_CORE_MASK).count("1"))))
    print(f"[FaceDetector] 混合架构：推理绑定 P 核 mask=0x{P_CORE_MASK:X}｜"
          f"torch 线程 {torch.get_num_threads()}")

# ===== MediaPipe 初始化 =====
def ascii_model_path(path):
    """MediaPipe 的 C++ 读取层不认中文路径：落到纯 ASCII 临时目录再喂给它。

    项目放在「主要项目」这类中文目录下时，直接 create_from_options 会报
    FileNotFoundError: Unable to open file，但文件其实是好的。
    """
    if all(ord(c) < 128 for c in path):
        return path
    import shutil
    import tempfile
    cache = os.path.join(tempfile.gettempdir(), "edulens_landmarker")
    os.makedirs(cache, exist_ok=True)
    dst = os.path.join(cache, "face_landmarker.task")
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(path):
        shutil.copy2(path, dst)
    return dst


base_options = python.BaseOptions(model_asset_path=ascii_model_path(landmarker_path))
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.3
)
landmarker = vision.FaceLandmarker.create_from_options(options)

if os.getenv("SECURITY_FORCE_CPU") == "1":
    device = torch.device("cpu")
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== MTCNN 初始化（降低阈值） =====
mtcnn = MTCNN(keep_all=False, device=device, thresholds=[0.4, 0.5, 0.5])

# ===== ResNet 初始化 =====
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


# ============================================================
# 特征提取函数（只使用 MTCNN，不再降级）
# ============================================================
#: 检测阶段的长边上限：哪怕摄像头驱动硬塞 1080p，也先缩到 640 再进 MTCNN。
#: 人脸最终被裁成 160×160 送 ResNet，多余分辨率对识别没有增益，只增加 CPU 耗时。
MAX_DETECT_SIDE = 640


def fit_for_detection(frame):
    h, w = frame.shape[:2]
    long_side = max(h, w)
    if long_side <= MAX_DETECT_SIDE:
        return frame
    scale = MAX_DETECT_SIDE / long_side
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def extract_face_feature(frame):
    """
    从单帧图像中提取人脸特征向量。
    只使用 MTCNN，失败返回 None。
    """
    sample = extract_face_sample(frame)
    return sample[0] if sample is not None else None


def extract_face_sample(frame, quality_gate=True):
    """检测 + 质量门控 + 提特征，MTCNN 只跑一次。

    返回 None 表示没检测到人脸；否则返回 (feature|None, QualityReport)。
    feature 为 None 说明这一帧质量不过关（糊、暗、侧脸、太远），调用方应换下一帧。
    quality_gate=False 时不做拒绝，专供评估脚本对比"有/无质量门控"的差异。
    """
    rgb = cv2.cvtColor(fit_for_detection(frame), cv2.COLOR_BGR2RGB)
    try:
        boxes, probs, landmarks = mtcnn.detect(rgb, landmarks=True)
    except Exception as e:
        print(f"[特征提取] MTCNN 出错: {e}")
        return None
    if boxes is None or len(boxes) == 0:
        return None

    index = int(np.argmax(probs)) if probs is not None else 0
    box = boxes[index]
    points = landmarks[index] if landmarks is not None else None
    height, width = rgb.shape[:2]

    try:
        face_tensor = mtcnn.extract(rgb, np.asarray([box], dtype=np.float32), None)
        if face_tensor is None:
            return None
        #: keep_all=False 时 extract 返回 (3,H,W)，补一维 batch 再送 ResNet
        if face_tensor.ndim == 3:
            face_tensor = face_tensor.unsqueeze(0)

        #: 质量评分统一在 MTCNN 对齐后的 160×160 人脸上做：不同来源的画面尺寸
        #: 差异很大（LFW 90px、摄像头 300px），在原始裁剪上算 Laplacian 方差
        #: 会让同一阈值在不同数据集上含义完全不同。
        aligned = face_tensor[0].permute(1, 2, 0).numpy() * 128.0 + 127.5
        aligned = np.clip(aligned, 0, 255).astype(np.uint8)
        report = assess_face(aligned, box=box, landmarks=points, frame_shape=(height, width))
        if quality_gate and not report.passed:
            return None, report

        face_tensor = face_tensor.to(device)
        with torch.no_grad():
            feature = resnet(face_tensor)[0].cpu().numpy()
    except Exception as e:
        print(f"[特征提取] 提特征失败: {e}")
        return None
    return feature, report


#: 注册/登录采集的帧数与最短时间预算。间隔 50ms 是为了拿到真正不同的帧，
#: 而不是同一帧被重复计入融合。
FUSION_SAMPLES = 8
FUSION_MIN_SAMPLES = 4
FUSION_TIMEOUT = 8.0


def collect_fusion_feature(camera, samples=FUSION_SAMPLES, min_samples=FUSION_MIN_SAMPLES,
                           timeout=FUSION_TIMEOUT, frontal=True, on_reject=None):
    """在摄像头上采集多帧通过质量门控的特征并融合。

    frontal=True 时额外要求近似正脸（yaw/roll 比质量门槛更严），避免把注册样本
    采成侧脸。返回 (融合特征|None, 有效帧数)。
    """
    captured = []
    deadline = time.time() + timeout
    while time.time() < deadline and len(captured) < samples:
        frame = camera.getframe()
        if frame is None:
            time.sleep(0.01)
            continue
        sample = extract_face_sample(frame)
        if sample is None:
            continue
        feature, report = sample
        if feature is None:
            if on_reject is not None:
                on_reject(report)
            time.sleep(0.05)
            continue
        if frontal and (report.yaw > 20 or report.roll > 15):
            if on_reject is not None:
                on_reject(report)
            time.sleep(0.05)
            continue
        captured.append(feature)
        time.sleep(0.05)

    if len(captured) < min_samples:
        return None, len(captured)
    return fuse_features(captured), len(captured)


def landmarks_from_frame(frame):
    """对单帧跑 FaceLandmarker，返回 468/478 个关键点；没脸返回 None。

    不挂在 userfaceinfo 上：那个类自带摄像头所有权，锁屏要复用界面上已经在跑的
    摄像头，只能走这个不持有设备的纯函数版本。
    """
    global _timestamp_counter
    rgb = cv2.cvtColor(fit_for_detection(frame), cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    #: detect_for_video 要求时间戳严格递增，掉帧也只能往前推
    _timestamp_counter += 10
    result = landmarker.detect_for_video(mp_image, _timestamp_counter)
    return result.face_landmarks[0] if result.face_landmarks else None


# ============================================================
# 摄像头封装（用于注册等独立场景）
# ============================================================
class userfaceinfo():
    def __init__(self, parent=None):
        self.parent = parent
        self.camera = camera_process()
        self.ifcamera = self.camera.start()

    def show_auto_close_message(self, text, duration):
        msg_box = QMessageBox(self.parent)
        msg_box.setWindowTitle("")
        msg_box.setText(text)
        msg_box.setStandardButtons(QMessageBox.NoButton)
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: rgba(20, 25, 50, 0.92);
                border-radius: 16px;
                border: 1px solid rgba(74, 158, 255, 0.3);
                padding: 10px;
            }
            QMessageBox QLabel {
                color: #e8edf5;
                font-size: 20px;
                font-weight: 500;
                font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
                padding: 20px 30px;
                min-width: 200px;
                min-height: 50px;
            }
            QMessageBox QPushButton {
                background-color: rgba(74, 158, 255, 0.2);
                border-radius: 8px;
                padding: 8px 24px;
                color: #4a9eff;
                font-size: 14px;
                font-weight: 600;
            }
            QMessageBox QPushButton:hover {
                background-color: rgba(74, 158, 255, 0.4);
            }
        """)
        if "👁️" in text or "眨眼" in text:
            msg_box.setWindowTitle("👁️ 活体检测")
            msg_box.setIcon(QMessageBox.Information)
        elif "👄" in text or "张嘴" in text:
            msg_box.setWindowTitle("👄 活体检测")
            msg_box.setIcon(QMessageBox.Information)
        elif "正对" in text or "👤" in text:
            msg_box.setWindowTitle("👤 活体检测")
            msg_box.setIcon(QMessageBox.Information)
        elif "✅" in text or "成功" in text:
            msg_box.setWindowTitle("✅ 完成")
            msg_box.setIcon(QMessageBox.Information)
        elif "❌" in text or "失败" in text or "超时" in text:
            msg_box.setWindowTitle("❌ 提示")
            msg_box.setIcon(QMessageBox.Warning)
        else:
            msg_box.setWindowTitle("提示")
            msg_box.setIcon(QMessageBox.Information)
        QTimer.singleShot(duration, msg_box.accept)
        msg_box.exec_()

    def get_landmarks(self, timeout=None):
        """获取一帧的人脸关键点（468 点）。

        timeout 秒内没检测到人脸就返回 None。不传 timeout 时保持原先的阻塞语义，
        等摄像头出人为止——注册流程要的正是"没脸就继续等，但别永远等下去"。
        """
        if not self.ifcamera:
            return None

        deadline = None if timeout is None else time.time() + timeout

        while True:
            frame = self.camera.getframe()
            if frame is None:
                if deadline is not None and time.time() >= deadline:
                    return None
                time.sleep(0.01)
                continue

            landmarks = landmarks_from_frame(frame)
            if landmarks is not None:
                return landmarks
            if deadline is not None and time.time() >= deadline:
                return None

    def get_frame(self):
        return self.camera.getframe()

    def release(self):
        self.camera.release()


# ============================================================
# 简化登录（跳过活体检测，直接特征提取）
# ============================================================
class simple_login_facedetector():
    def __init__(self):
        pass

    def simple_login(self, camera=None):
        own_camera = False
        if camera is None:
            self.facedetector = userfaceinfo()
            if not self.facedetector.ifcamera:
                return False, None
            camera = self.facedetector.camera
            own_camera = True
        else:
            self.facedetector = None

        max_duration = 10.0
        feature, count = collect_fusion_feature(camera, timeout=max_duration)

        if own_camera:
            camera.release()

        if feature is not None:
            print(f"[简化登录] 融合 {count} 帧通过质量门控的人脸特征")
            return True, feature
        print(f"[简化登录] 超时或人脸质量不足（有效帧 {count}）")
        return False, None


# ============================================================
# 活体挑战表：顺序每轮打乱，四个全过才算真人（四步随机状态机）
# ============================================================
@dataclass(frozen=True)
class LivenessChallenge:
    key: str
    prompt: str
    settle: float
    timeout_msg: str


LIVENESS_CHALLENGES = (
    LivenessChallenge("blink", "👁️ 请眨眼", 0.3, "❌ 眨眼检测超时"),
    LivenessChallenge("mouth", "👄 请张嘴", 1.5, "❌ 张嘴检测超时"),
    LivenessChallenge("pose", "👤 请正对摄像头", 1.5, "❌ 正脸检测超时"),
    LivenessChallenge("motion", "⏳ 请稍等，检测中...", 1.5, "❌ 微动检测超时"),
)


# ============================================================
# 注册流程
# ============================================================
class register_face_detector():
    def __init__(self, blink_thresh=0.25, mouth_thresh=0.04, yaw_thresh=60, motion_thresh=0.01):
        self.blink_thresh = blink_thresh
        self.mouth_thresh = mouth_thresh
        self.yaw_thresh = yaw_thresh
        self.motion_thresh = motion_thresh
        self.prev_landmarks = None

        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 387, 385, 263, 373, 380]
        self.MOUTH = [61, 78, 13, 14, 308, 291]
        self.NOSE_TIP = 1
        self.LEFT_EYE_OUTER = 33
        self.RIGHT_EYE_OUTER = 133

        self.facedetector = None

    def _get_points(self, landmarks, indices):
        return [(landmarks[i].x, landmarks[i].y) for i in indices]

    def _compute_ratio(self, points):
        p1, p2, p3, p4, p5, p6 = points
        vert1 = math.hypot(p2[0]-p6[0], p2[1]-p6[1])
        vert2 = math.hypot(p3[0]-p5[0], p3[1]-p5[1])
        horiz = math.hypot(p1[0]-p4[0], p1[1]-p4[1])
        if horiz == 0:
            return 0.0
        return (vert1 + vert2) / (2.0 * horiz)

    def check_blink(self, landmarks):
        left = self._get_points(landmarks, self.LEFT_EYE)
        right = self._get_points(landmarks, self.RIGHT_EYE)
        ear = (self._compute_ratio(left) + self._compute_ratio(right)) / 2.0
        return ear < self.blink_thresh

    def check_mouth_open(self, landmarks):
        mouth = self._get_points(landmarks, self.MOUTH)
        mar = self._compute_ratio(mouth)
        return mar > self.mouth_thresh

    def check_head_pose(self, landmarks):
        nose = landmarks[self.NOSE_TIP]
        left = landmarks[self.LEFT_EYE_OUTER]
        right = landmarks[self.RIGHT_EYE_OUTER]
        eye_center_x = (left.x + right.x) / 2.0
        dx = nose.x - eye_center_x
        width = abs(right.x - left.x)
        if width < 1e-6:
            return False
        yaw_angle = abs(dx) / width * 60
        return yaw_angle < self.yaw_thresh

    def check_motion(self, landmarks):
        """连续两帧关键点位移超过阈值才算"微动"。

        第一帧只留作基线、不算通过——以前返回 True，挑战排到第一位时会白送一分，
        打乱顺序以后这个漏洞每次都会露出来。
        """
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            return False
        key_idx = [1, 33, 133, 61, 291]
        cur = np.array([(landmarks[i].x, landmarks[i].y) for i in key_idx])
        prev = np.array([(self.prev_landmarks[i].x, self.prev_landmarks[i].y) for i in key_idx])
        dist = np.mean(np.linalg.norm(cur - prev, axis=1))
        self.prev_landmarks = landmarks
        return bool(dist > self.motion_thresh)

    def liveness_order(self):
        """本轮挑战顺序：四个全要，顺序打乱（简历里的四步随机状态机）。"""
        order = list(LIVENESS_CHALLENGES)
        random.shuffle(order)
        return order

    def _run_check(self, key, landmarks):
        return {
            "blink": self.check_blink,
            "mouth": self.check_mouth_open,
            "pose": self.check_head_pose,
            "motion": self.check_motion,
        }[key](landmarks)

    def _release_camera(self):
        cam = getattr(self.facedetector, "camera", None)
        if cam is not None:
            try:
                cam.release()
            except Exception as e:
                print(f"[注册] 释放摄像头失败: {e}")
        cv2.destroyAllWindows()

    def ifpass(self, detector=None, challenges=None, timeout=20.0):
        """活体 + 人脸采集。

        四个挑战按随机顺序逐个做，每个 challenge 单独给 timeout 秒；
        全部通过才采集特征。detector/challenges 是给无摄像头自检留的注入口。
        """
        self.facedetector = detector or userfaceinfo()
        if not self.facedetector.ifcamera:
            return False, None

        order = list(challenges) if challenges is not None else self.liveness_order()
        print("[活体] 本轮挑战顺序:", " → ".join(c.key for c in order))

        for ch in order:
            self.facedetector.show_auto_close_message(ch.prompt, 3000)
            time.sleep(ch.settle)
            #: 微动要拿本次挑战内的两帧比，不能沿用上一轮留下的关键点当基线
            self.prev_landmarks = None
            deadline = time.time() + timeout
            passed = False
            while time.time() < deadline:
                landmarks = self.facedetector.get_landmarks(timeout=0.5)
                if landmarks is None:
                    continue
                if self._run_check(ch.key, landmarks):
                    passed = True
                    break
            if not passed:
                self.facedetector.show_auto_close_message(ch.timeout_msg, 3000)
                self._release_camera()
                return False, None

        try:
            faceinfo = self.hand_in(True)
        except Exception as e:
            print(f"[注册] hand_in 异常: {e}")
            self._release_camera()
            return False, None

        self._release_camera()

        if faceinfo is not None:
            return True, faceinfo
        return False, None

    def hand_in(self, is_auto):
        if not self.facedetector or not self.facedetector.ifcamera:
            return None
        feature, count = collect_fusion_feature(
            self.facedetector.camera, samples=10, min_samples=4, timeout=8.0
        )
        if feature is not None:
            print(f"[注册] 融合 {count} 帧通过质量门控的人脸特征")
            return feature
        self.facedetector.show_auto_close_message(
            "人脸质量不足，请保持正脸、光线充足并靠近镜头", 2500
        )
        return None


def run_liveness(camera, prompt=None, timeout=8.0, challenges=None, landmark_provider=None):
    """在指定摄像头上跑四步分步挑战式活体，四个全过返回 True。

    和注册流程共用判据与随机顺序，区别只有两点：摄像头由调用方持有（锁屏复用它
    已经在跑的相机），提示走 prompt 回调而不是模态弹窗。timeout 是每个挑战的
    绝对时间预算，不随帧率漂移。
    """
    detector = register_face_detector()
    order = list(challenges) if challenges is not None else detector.liveness_order()
    read_landmarks = landmark_provider or landmarks_from_frame
    print("[活体] 本轮挑战顺序:", " → ".join(c.key for c in order))

    for ch in order:
        if prompt is not None:
            prompt(ch.prompt)
        time.sleep(ch.settle)
        #: 微动要拿本次挑战内的两帧比，不能沿用上一轮留下的关键点当基线
        detector.prev_landmarks = None
        deadline = time.time() + timeout
        passed = False
        while time.time() < deadline:
            frame = camera.getframe()
            if frame is None:
                time.sleep(0.01)
                continue
            landmarks = read_landmarks(frame)
            if landmarks is None:
                continue
            if detector._run_check(ch.key, landmarks):
                passed = True
                break
        if not passed:
            if prompt is not None:
                prompt(ch.timeout_msg)
            return False
    return True


# ============================================================
# auto_hand_controller 类
# ============================================================
class auto_hand_controller():
    def __init__(self):
        pass

    def run(self, isauto, camera=None):
        if isauto == False:
            detector = register_face_detector()
            result = detector.ifpass()
            if result is None or not isinstance(result, tuple) or len(result) != 2:
                return False, None
            return result[0], result[1]
        else:
            print("[自动登录] 提取人脸特征（活体检测由调用方执行）")
            simple_det = simple_login_facedetector()
            success, feature = simple_det.simple_login(camera=camera)
            return success, feature