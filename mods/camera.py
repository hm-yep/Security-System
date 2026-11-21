import cv2
import threading
import time

class camera_process:
    _instance = None          # 单例实例
    _lock = threading.Lock()  # 单例锁

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 初始化只执行一次（通过 hasattr 判断）
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        self.cap = None
        self.frame = None
        self.running = False
        self.thread = None
        self.frame_lock = threading.Lock()   # 帧锁

    def start(self):
        """打开摄像头并启动后台读取线程（单例模式下只启动一次）"""
        if self.cap is not None and self.cap.isOpened():
            return True          # 已经打开

        self.cap = cv2.VideoCapture(0)
        #: 640×480 足够：人脸最终被裁成 160×160 送进 ResNet，1080p 只会让 MTCNN
        #: 在 CPU 上多花几倍时间（实测 720p 单帧 0.4s+，640×480 约 0.2s）。
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            self.cap = None
            return False

        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

        # 等待第一帧
        wait_time = 0
        while self.frame is None and wait_time < 2.0:
            time.sleep(0.05)
            wait_time += 0.05

        return self.frame is not None

    def _update_loop(self):
        """后台线程持续读取帧，存入 self.frame"""
        while self.running and self.cap is not None:
            success, frame = self.cap.read()
            if success:
                with self.frame_lock:
                    self.frame = frame
            else:
                time.sleep(0.01)   # 读取失败稍作等待

    def getframe(self):
        """获取最新帧（线程安全）"""
        with self.frame_lock:
            return self.frame

    def cap_open(self):
        """兼容旧接口"""
        return self.cap is not None and self.cap.isOpened()

    def getframehw(self):
        frame = self.getframe()
        if frame is None:
            return None
        height, width, _ = frame.shape
        return height, width

    def release(self):
        """释放摄像头（单例模式下通常不主动释放，程序退出时自动清理）"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        with self.frame_lock:
            self.frame = None