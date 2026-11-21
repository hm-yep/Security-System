# Security System · 基于人脸与活体检测的 Windows 桌面安防系统

一个端到端的单机桌面应用：摄像头采集 → 人脸检测与 512 维特征比对 → 四步随机活体挑战 → 自动锁屏 / 解锁，配套注册登录、审计日志、开机自启与系统托盘。

> 定位说明：这是一个**工程实现型**项目，重点是多线程、CPU 推理性能调优、Windows 系统层交互与端到端闭环；模型均为成熟预训练模型，未做训练或微调。

## 功能

- 注册 / 登录 / 改密 / 注销、人脸录入与更换、账户切换
- 四步随机顺序活体检测（眨眼、张嘴、正脸、微动），全部通过才放行
- 空闲检测自动锁屏、陌生人停留告警、Windows API 系统锁屏兜底
- 锁屏期间低级键鼠钩子拦截（Win / Alt+Tab / Alt+F4 / Ctrl+Esc 等）
- 审计日志查询界面、注册表开机自启、系统托盘常驻

## 模块结构

| 文件 | 职责 |
| --- | --- |
| `main.py` | 入口，按 `--auto` 选择手动界面或锁屏 |
| `mods/face_detector.py` | MTCNN + InceptionResnetV1 提特征、MediaPipe 478 点、EAR/MAR/姿态/微动判据、P 核绑定 |
| `mods/auto_ui.py` | 锁屏窗口、QThread 检测线程、InputGuard 键鼠钩子、YOLOv8 人体检测与陌生人告警 |
| `mods/hand_ui.py` | 登录/注册/主界面、托盘、空闲检测、日志界面 |
| `mods/database.py` | MySQL 数据访问层、口令哈希与渐进迁移、审计日志、在线状态 |
| `mods/feature_codec.py` | 512 维特征的序列化与只读解析 |
| `mods/security.py` | PBKDF2-HMAC-SHA256 口令哈希 |
| `mods/config.py` | 环境变量 / `.env` 配置读取 |
| `mods/camera.py` | 摄像头单例、后台读帧线程、帧锁 |
| `mods/startup_manager.py` | 注册表自启项管理 |
| `verify_claims.py` | 无摄像头注入式自检，逐条验收功能声明 |
| `bench_cpu_latency.py` | CPU 单帧推理耗时实测（最快/中位/均值/P95） |
| `tests/` | 安全与编解码回归测试（不需要摄像头和数据库） |

## 快速开始

环境：Windows + Python 3.10~3.13 + 本机 MySQL + 可用摄像头。`mediapipe` 对较新的 Python 版本可能没有预编译包，遇到安装问题建议用 3.10/3.11。

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # 只需跑测试时
cp .env.example .env                 # Windows: copy .env.example .env
```

在 `.env` 里填写本机数据库口令（该文件已被 `.gitignore` 排除，不要提交）。先建库，表由程序启动时自建：

```sql
CREATE DATABASE IF NOT EXISTS Security_System_Users_DataBase
  DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

模型文件放在项目根目录（仓库已包含）：`yolov8n.pt`、`face_landmarker.task`、`facenet_pytorch_data/{pnet,rnet,onet}.pt`。MediaPipe 不认中文路径，程序会自动把模型复制到纯 ASCII 临时目录再加载。

```bash
python main.py            # 手动登录界面
python main.py --auto     # 直接进入锁屏（开机自启用的就是这个）
```

验证：

```bash
pytest -q tests/                 # 安全与编解码回归，不需要摄像头/数据库/模型
python verify_claims.py          # 逐条验收功能声明（会短暂安装并摘除键鼠钩子）
python bench_cpu_latency.py      # CPU 单帧推理基准，口径 640x480
```

## 设计要点

- **CPU 推理调优**：混合架构上用 `GetLogicalProcessorInformationEx` 找出 P 核并设置进程亲和性（E 核实测慢 4~5 倍），torch 线程封顶 8；检测前把长边缩到 640（人脸最终裁成 160×160，更高分辨率没有识别增益）。`bench_cpu_latency.py` 给出可复现的 p50/p95。
- **活体状态机**：四个挑战每轮随机排序、每个挑战独立超时预算；微动判据第一帧只做基线，避免挑战排首位时"白送一分"；身份匹配通过后仍必须过活体，拦截照片/静态图像。
- **线程模型**：摄像头单例 + 后台读帧线程 + 帧锁；识别放在 `QThread`，通过信号回主线程更新 UI。
- **锁屏双层**：自定义无边框置顶窗口负责刷脸解锁，构建失败或用户主动要求时回退到 `LockWorkStation` 交给 Windows 安全桌面。
- **凭据安全**：数据库口令只从环境变量 / `.env` 读取；口令用 PBKDF2-HMAC-SHA256 + 随机盐存储，历史明文口令在登录成功后自动逐行升级（无需停服刷库），同时幂等把 `password` 列从 `VARCHAR(20)` 扩到 `VARCHAR(255)`；特征串解析只用 `json` / `literal_eval` / 数值扫描，不执行数据库内容；SQL 全参数化。

## 已知限制

- 仅 Windows 可运行，强依赖本机 MySQL、摄像头和随仓库附带的模型权重，面试官无法一键复现（建议配一段演示录屏）。
- 模型全部是预训练调包，没有训练、微调或自采数据集；识别阈值 0.6 为经验值，**尚未测过 FAR/FRR**，活体也未做视频回放/面具攻击的实测。
- UI 层仍残留少量直接 SQL（`enroll_face` / `view_logs` / 空闲检测等），分层不彻底；数据库连接未池化，异常提示仍以弹窗为主。
- `userwindowpassword` 字段目前无业务使用，保留仅为兼容。
- 演示材料待补：30 秒操作录屏、照片攻击被拦截的 GIF、benchmark 实测数值表。

## 许可证与说明

仅用于学习与演示，不要作为生产级身份认证系统使用。
