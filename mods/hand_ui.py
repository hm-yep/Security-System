# mods/ui.py
from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QMessageBox, QFrame,
    QGraphicsDropShadowEffect, QApplication, QDialog,
    QSystemTrayIcon, QMenu, QAction
)
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, QTimer
from PyQt5.QtGui import QColor, QIcon
from .database import package
from .feature_codec import serialize_feature, parse_feature as decode_feature
from .face_detector import auto_hand_controller, mtcnn, resnet, device
from .startup_manager import is_startup_enabled, enable_startup, disable_startup
from .camera import camera_process
import ctypes
from ctypes import wintypes
import time
import cv2
import torch
import numpy as np
import sys
class FadeMessageBox(QWidget):
    def __init__(self, parent=None, message="", duration=1000):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("""
            QLabel#msgLabel {
                background-color: rgba(20, 30, 50, 0.85);
                color: #00ff88;
                font-size: 28px;
                font-weight: 600;
                border-radius: 20px;
                padding: 30px 50px;
                border: 1px solid rgba(0, 255, 136, 0.2);
            }
        """)
        self.label = QLabel(message, self)
        self.label.setObjectName("msgLabel")
        self.label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.adjustSize()
        if parent:
            parent_rect = parent.geometry()
            x = parent_rect.x() + (parent_rect.width() - self.width()) // 2
            y = parent_rect.y() + (parent_rect.height() - self.height()) // 2
            self.move(x, y)
        else:
            screen = QApplication.primaryScreen().geometry()
            self.move((screen.width() - self.width()) // 2, (screen.height() - self.height()) // 2)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.start_fade_out)
        self.timer.start(duration)
        self.fade_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_animation.setDuration(400)
        self.fade_animation.setStartValue(1.0)
        self.fade_animation.setEndValue(0.0)
        self.fade_animation.setEasingCurve(QEasingCurve.OutQuad)
        self.fade_animation.finished.connect(self.close)
        self.label.mousePressEvent = lambda e: self.close()

    def start_fade_out(self):
        self.fade_animation.start()


class InputDialog(QDialog):
    def __init__(self, parent=None, title="", label1="", label2="",
                 placeholder1="", placeholder2="", is_password=False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(380)
        self.setStyleSheet("""
            QDialog { background-color: #0d0d1a; }
            QLabel { color: #e8edf5; font-size: 14px; font-weight: 500; }
            QLineEdit {
                background-color: #1a1a32;
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 10px;
                padding: 10px 14px;
                color: #e8edf5;
                font-size: 14px;
            }
            QLineEdit:focus { border-color: #4a9eff; }
            QPushButton {
                background-color: #4a9eff;
                color: #ffffff;
                font-weight: 600;
                border: none;
                border-radius: 10px;
                padding: 10px 28px;
                font-size: 14px;
                min-width: 80px;
            }
            QPushButton:hover { background-color: #5aaeff; }
            QPushButton#cancelBtn {
                background-color: transparent;
                border: 1px solid rgba(255,255,255,0.1);
                color: #aab8d0;
            }
            QPushButton#cancelBtn:hover { background-color: rgba(255,255,255,0.05); }
        """)
        layout = QVBoxLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(30, 30, 30, 30)
        if label1:
            lbl1 = QLabel(label1)
            layout.addWidget(lbl1)
            self.input1 = QLineEdit()
            self.input1.setPlaceholderText(placeholder1)
            if is_password:
                self.input1.setEchoMode(QLineEdit.Password)
            layout.addWidget(self.input1)
        if label2:
            lbl2 = QLabel(label2)
            layout.addWidget(lbl2)
            self.input2 = QLineEdit()
            self.input2.setPlaceholderText(placeholder2)
            if is_password:
                self.input2.setEchoMode(QLineEdit.Password)
            layout.addWidget(self.input2)
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        self.ok_btn = QPushButton("确认")
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setObjectName("cancelBtn")
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

    def get_inputs(self):
        if hasattr(self, 'input1') and hasattr(self, 'input2'):
            return self.input1.text(), self.input2.text()
        elif hasattr(self, 'input1'):
            return self.input1.text(),
        return ()


class ConfirmDialog(QDialog):
    def __init__(self, parent=None, title="确认", message="确定执行此操作吗？",
                 icon_type="warning"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(350)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setStyleSheet("""
            QDialog { background-color: #0d0d1a; }
            QLabel { color: #e8edf5; font-size: 15px; padding: 10px; }
            QPushButton#yesBtn {
                background-color: #4a9eff; color: #ffffff;
                font-weight: 600; border: none; border-radius: 10px;
                padding: 10px 28px; font-size: 14px; min-width: 80px;
            }
            QPushButton#yesBtn:hover { background-color: #5aaeff; }
            QPushButton#noBtn {
                background-color: transparent;
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 10px; padding: 10px 28px;
                color: #aab8d0; font-size: 14px; min-width: 80px;
            }
            QPushButton#noBtn:hover { background-color: rgba(255,255,255,0.05); }
            QPushButton#dangerBtn {
                background-color: #ff4757; color: #ffffff;
                font-weight: 600; border: none; border-radius: 10px;
                padding: 10px 28px; font-size: 14px; min-width: 80px;
            }
            QPushButton#dangerBtn:hover { background-color: #ff6b7a; }
        """)
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(30, 25, 30, 25)
        msg_layout = QHBoxLayout()
        icon_label = QLabel("⚠️" if icon_type == "warning" else "❓")
        icon_label.setStyleSheet("font-size: 32px;")
        msg_layout.addWidget(icon_label)
        msg_label = QLabel(message)
        msg_label.setWordWrap(True)
        msg_layout.addWidget(msg_label, 1)
        layout.addLayout(msg_layout)
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        self.yes_btn = QPushButton("确认")
        if icon_type == "danger":
            self.yes_btn.setObjectName("dangerBtn")
        else:
            self.yes_btn.setObjectName("yesBtn")
        self.no_btn = QPushButton("取消")
        self.no_btn.setObjectName("noBtn")
        btn_layout.addStretch()
        btn_layout.addWidget(self.yes_btn)
        btn_layout.addWidget(self.no_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.yes_btn.clicked.connect(self.accept)
        self.no_btn.clicked.connect(self.reject)


LOGI_STYLE = """
QWidget {
    background-color: #0d0d1a;
    color: #e8edf5;
    font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif;
}
QFrame#card {
    background-color: #16162e;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.04);
}
QFrame#card:hover {
    background-color: #1e1e3a;
    border-color: rgba(255, 255, 255, 0.08);
}
QLineEdit {
    background-color: #1a1a32;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 12px 16px;
    color: #e8edf5;
    font-size: 14px;
}
QLineEdit:focus { border-color: #4a9eff; background-color: #1e1e3a; }
QLineEdit:hover { border-color: rgba(255, 255, 255, 0.12); }
QPushButton#primaryBtn {
    background-color: #4a9eff; color: #ffffff; font-weight: 600;
    border: none; border-radius: 10px; padding: 12px 28px; font-size: 14px;
}
QPushButton#primaryBtn:hover { background-color: #5aaeff; }
QPushButton#primaryBtn:pressed { background-color: #3a8eef; }
QPushButton#secondaryBtn {
    background-color: transparent;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 10px; padding: 12px 28px;
    color: #aab8d0; font-size: 14px;
}
QPushButton#secondaryBtn:hover {
    background-color: rgba(255, 255, 255, 0.04);
    border-color: rgba(255, 255, 255, 0.18);
    color: #ffffff;
}
QLabel#titleLabel { font-size: 28px; font-weight: 300; color: #ffffff; }
QLabel#subtitleLabel { font-size: 14px; color: rgba(255, 255, 255, 0.35); }
QLabel#greetingLabel { font-size: 32px; font-weight: 300; color: #ffffff; }
QLabel#statusLabel { font-size: 14px; color: #4a9eff; }
QLabel#cardTitle { font-size: 15px; font-weight: 600; color: #ffffff; }
QLabel#cardDesc { font-size: 13px; color: rgba(255, 255, 255, 0.35); }
"""


class loginwindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Security System")
        self.resize(520, 650)
        self.setMinimumSize(420, 550)
        self.setStyleSheet(LOGI_STYLE)
        self.setWindowIcon(QIcon("logo/logo.png"))
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(50, 60, 50, 50)
        main_layout.setSpacing(0)

        title = QLabel("🔐 Security System")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        subtitle = QLabel("人脸识别安全登录")
        subtitle.setObjectName("subtitleLabel")
        subtitle.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle)
        main_layout.addSpacing(50)

        card = QFrame()
        card.setObjectName("card")
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 50))
        card.setGraphicsEffect(shadow)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(35, 35, 35, 35)
        card_layout.setSpacing(18)

        lbl_user = QLabel("账号")
        lbl_user.setStyleSheet("color: rgba(255,255,255,0.5); font-size: 13px;")
        self.line_username = QLineEdit()
        self.line_username.setPlaceholderText("请输入用户名")
        self.line_username.setFixedHeight(48)

        lbl_pwd = QLabel("密码")
        lbl_pwd.setStyleSheet("color: rgba(255,255,255,0.5); font-size: 13px;")
        self.line_userpw = QLineEdit()
        self.line_userpw.setPlaceholderText("请输入密码")
        self.line_userpw.setEchoMode(QLineEdit.Password)
        self.line_userpw.setFixedHeight(48)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)
        self.btn_login = QPushButton("登 录")
        self.btn_login.setObjectName("primaryBtn")
        self.btn_login.setFixedHeight(50)
        self.btn_register = QPushButton("注 册")
        self.btn_register.setObjectName("secondaryBtn")
        self.btn_register.setFixedHeight(50)

        btn_layout.addWidget(self.btn_login)
        btn_layout.addWidget(self.btn_register)

        card_layout.addWidget(lbl_user)
        card_layout.addWidget(self.line_username)
        card_layout.addWidget(lbl_pwd)
        card_layout.addWidget(self.line_userpw)
        card_layout.addSpacing(15)
        card_layout.addLayout(btn_layout)

        main_layout.addWidget(card)
        main_layout.addStretch()
        self.setLayout(main_layout)

        self.btn_login.clicked.connect(self.connecthandle_login)
        self.btn_register.clicked.connect(self.connecthandle_register)

    def connecthandle_login(self):
        username = self.line_username.text()
        password = self.line_userpw.text()
        global UN
        UN = username
        if username == "":
            QMessageBox.information(self, "提示", "用户名不能为空！")
            return
        elif password == "":
            QMessageBox.information(self, "提示", "密码不能为空！")
            return
        is_empty = package().is_empty()
        if is_empty:
            QMessageBox.information(self, "提示", "请先完成注册！")
            return
        if package().if_info_exist(username, password):
            pkg = package()
            pkg.insertdatalog(f"用户 {username} 登录系统", username, None)
            pkg.insertinfo_to_current_user(None, username, "online", None, None)
            self.fade_msg = FadeMessageBox(self, message="✅ 欢迎回来 🎉", duration=1000)
            self.fade_msg.show()
            QTimer.singleShot(300, self.open_main_window)
        else:
            QMessageBox.information(self, "提示", "用户名或密码错误")

    def open_main_window(self):
        self.MainWindow = mainwindow()
        self.MainWindow.show()
        self.close()

    def connecthandle_register(self):
        self.register_ui = registerwindow()
        self.register_ui.show()


class registerwindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Security System Register UI")
        self.resize(520, 650)
        self.setMinimumSize(420, 550)
        self.setStyleSheet(LOGI_STYLE)
        self.setWindowIcon(QIcon("logo/logo.png"))
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(50, 60, 50, 50)
        main_layout.setSpacing(0)

        title = QLabel("📝 注册新账号")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        subtitle = QLabel("填写信息并完成人脸采集")
        subtitle.setObjectName("subtitleLabel")
        subtitle.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle)
        main_layout.addSpacing(50)

        card = QFrame()
        card.setObjectName("card")
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 50))
        card.setGraphicsEffect(shadow)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(35, 35, 35, 35)
        card_layout.setSpacing(18)

        lbl_user = QLabel("账号")
        lbl_user.setStyleSheet("color: rgba(255,255,255,0.5); font-size: 13px;")
        self.line_username = QLineEdit()
        self.line_username.setPlaceholderText("设置用户名")
        self.line_username.setFixedHeight(48)

        lbl_pwd = QLabel("密码")
        lbl_pwd.setStyleSheet("color: rgba(255,255,255,0.5); font-size: 13px;")
        self.line_userpw = QLineEdit()
        self.line_userpw.setPlaceholderText("设置密码")
        self.line_userpw.setEchoMode(QLineEdit.Password)
        self.line_userpw.setFixedHeight(48)

        self.btn_register = QPushButton("开始注册")
        self.btn_register.setObjectName("primaryBtn")
        self.btn_register.setFixedHeight(50)

        hint = QLabel("💡 注册将采集人脸信息用于身份验证")
        hint.setStyleSheet("color: rgba(255,255,255,0.3); font-size: 13px;")
        hint.setAlignment(Qt.AlignCenter)

        card_layout.addWidget(lbl_user)
        card_layout.addWidget(self.line_username)
        card_layout.addWidget(lbl_pwd)
        card_layout.addWidget(self.line_userpw)
        card_layout.addSpacing(15)
        card_layout.addWidget(self.btn_register)
        card_layout.addWidget(hint)

        main_layout.addWidget(card)
        main_layout.addStretch()
        self.setLayout(main_layout)

        self.btn_register.clicked.connect(self.connecthandle_register_ui)

    def connecthandle_register_ui(self):
        username = self.line_username.text().strip()
        password = self.line_userpw.text().strip()
        if not username or not password:
            QMessageBox.information(self, "提示", "用户名或密码不能为空")
            return

        pkg = package()
        if pkg.if_username_exist(username) and not pkg.is_empty():
            QMessageBox.information(self, "提示", "用户名已存在，请换一个")
            return

        cam_test = camera_process()
        if not cam_test.start():
            QMessageBox.warning(self, "错误", "无法打开摄像头，无法采集人脸信息！")
            return
        cam_test.release()

        QMessageBox.information(self, "提示", "接下来要采集人脸信息，请摆出正脸")
        userfaceclass = auto_hand_controller()
        ifpass, userfaceinfo = userfaceclass.run(False)
        if ifpass == False or userfaceinfo is None:
            QMessageBox.information(self, "提示", "获取人脸信息失败，请重试")
            return

        try:
            face_json = serialize_feature(userfaceinfo)
            pkg.getinfo(username, password, face_json, 0)
            pkg.insertdatalog(f"新用户 {username} 注册成功", username, None)
            QMessageBox.information(self, "成功", "注册成功！")
            self.close()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存用户信息失败: {e}")


def lock_workstation_api(api=None):
    """调用 Windows API 锁定当前会话（等价 Win+L）。

    api 是给无副作用自检留的注入口：真调一次会把演示机锁掉，
    验收脚本只需要确认"确实走到了这个 API"。
    """
    call = api or ctypes.windll.user32.LockWorkStation
    return bool(call())


class mainwindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.username = UN
        self.setWindowTitle("Security_System - 个人中心")
        self.resize(1100, 750)
        self.setMinimumSize(900, 650)
        self.setStyleSheet(LOGI_STYLE)
        self.setWindowIcon(QIcon("logo/logo.png"))

        self.tray_icon = None
        self.setup_tray()

        self.lock_screen_active = False
        self.lock_screen_window = None

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(70, 55, 70, 55)
        main_layout.setSpacing(35)

        top_layout = QHBoxLayout()
        top_layout.setSpacing(0)

        greeting_layout = QVBoxLayout()
        greeting_layout.setSpacing(4)
        greeting = QLabel(f"下午好，{self.username}")
        greeting.setObjectName("greetingLabel")
        greeting_layout.addWidget(greeting)
        status = QLabel("✅ 安全监控已开启")
        status.setObjectName("statusLabel")
        greeting_layout.addWidget(status)

        top_layout.addLayout(greeting_layout)
        top_layout.addStretch()

        self.btn_exit = QPushButton("✕ 退出程序")
        self.btn_exit.setObjectName("secondaryBtn")
        self.btn_exit.setFixedHeight(50)
        self.btn_exit.setFixedWidth(150)
        top_layout.addWidget(self.btn_exit)

        main_layout.addLayout(top_layout)

        grid_layout = QVBoxLayout()
        grid_layout.setSpacing(18)

        row1 = QHBoxLayout()
        row1.setSpacing(18)
        row1.addWidget(self.create_card("🔑", "修改密码", "更新当前登录密码"))
        row1.addWidget(self.create_card("📋", "获取操作日志", "查看所有操作记录"))
        grid_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(18)
        row2.addWidget(self.create_card("📸", "录入人脸信息", "首次录入人脸信息"))
        row2.addWidget(self.create_card("🔄", "更换人脸信息", "重新采集人脸信息"))
        grid_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.setSpacing(18)
        row3.addWidget(self.create_card("🔄", "切换账户", "切换到其他用户"))
        row3.addWidget(self.create_card("🗑️", "注销账户", "永久删除当前账户"))
        grid_layout.addLayout(row3)

        row4 = QHBoxLayout()
        row4.setSpacing(18)
        row4.addWidget(self.create_card("⚙️", "开机自启动", "设置是否开机自动运行"))
        row4.addWidget(self.create_card("🔒", "立即锁屏", "手动锁定系统（人脸解锁）"))
        grid_layout.addLayout(row4)

        main_layout.addLayout(grid_layout)
        main_layout.addStretch()
        self.setLayout(main_layout)

        self.btn_exit.clicked.connect(self.close)

        self.idle_timer = QTimer(self)
        self.idle_timer.timeout.connect(self.check_idle_and_scan)
        self.idle_timer.start(5000)
        self.idle_check_pending = False

        self.quick_lock_timer = QTimer(self)
        self.quick_lock_timer.setSingleShot(True)
        self.quick_lock_timer.timeout.connect(self.quick_lock_check)
        self.quick_lock_triggered = False
        QTimer.singleShot(3500, self.start_quick_lock_check)

    def setup_tray(self):
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon('logo/logo.png'))
        tray_menu = QMenu()
        show_action = QAction("🖥️ 显示主窗口", self)
        show_action.triggered.connect(self.show_main_window)
        tray_menu.addAction(show_action)
        tray_menu.addSeparator()
        self.auto_start_action = QAction("🔓 开机自启动", self)
        self.auto_start_action.setCheckable(True)
        self.auto_start_action.setChecked(is_startup_enabled())
        self.auto_start_action.triggered.connect(self.toggle_auto_start)
        tray_menu.addAction(self.auto_start_action)
        tray_menu.addSeparator()
        lock_action = QAction("🔒 系统锁屏（Windows API）", self)
        lock_action.triggered.connect(lambda: self.lock_workstation())
        tray_menu.addAction(lock_action)
        tray_menu.addSeparator()
        exit_action = QAction("✕ 退出程序", self)
        exit_action.triggered.connect(self.exit_application)
        tray_menu.addAction(exit_action)
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self.on_tray_activated)
        self.tray_icon.show()

    def show_main_window(self):
        if self.lock_screen_active:
            return
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.DoubleClick:
            self.show_main_window()

    def toggle_auto_start(self, checked):
        if checked:
            if not enable_startup():
                QMessageBox.warning(self, "错误", "开启开机自启动失败！")
                self.auto_start_action.setChecked(False)
        else:
            if not disable_startup():
                QMessageBox.warning(self, "错误", "关闭开机自启动失败！")
                self.auto_start_action.setChecked(True)

    def toggle_auto_start_from_card(self):
        current_state = is_startup_enabled()
        if current_state:
            if disable_startup():
                QMessageBox.information(self, "提示", "开机自启动已关闭")
                self.auto_start_action.setChecked(False)
            else:
                QMessageBox.warning(self, "错误", "关闭开机自启动失败！")
        else:
            if enable_startup():
                QMessageBox.information(self, "提示", "开机自启动已开启")
                self.auto_start_action.setChecked(True)
            else:
                QMessageBox.warning(self, "错误", "开启开机自启动失败！")

    def exit_application(self):
        if self.tray_icon:
            self.tray_icon.hide()
        QApplication.quit()

    def create_card(self, icon, title, desc):
        card = QFrame()
        card.setObjectName("card")
        card.setFixedHeight(120)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 40))
        card.setGraphicsEffect(shadow)
        layout = QHBoxLayout(card)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(18)
        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet("font-size: 30px;")
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        title_lbl = QLabel(title)
        title_lbl.setObjectName("cardTitle")
        desc_lbl = QLabel(desc)
        desc_lbl.setObjectName("cardDesc")
        text_layout.addWidget(title_lbl)
        text_layout.addWidget(desc_lbl)
        arrow = QLabel("›")
        arrow.setStyleSheet("font-size: 26px; color: rgba(255,255,255,0.08);")
        layout.addWidget(icon_lbl)
        layout.addLayout(text_layout, 1)
        layout.addWidget(arrow)
        card.mousePressEvent = lambda e: self.on_card_clicked(title)
        return card

    def on_card_clicked(self, title):
        if title == "修改密码":
            self.change_password()
        elif title == "获取操作日志":
            self.view_logs()
        elif title == "录入人脸信息":
            self.enroll_face()
        elif title == "更换人脸信息":
            self.update_face()
        elif title == "切换账户":
            self.switch_account()
        elif title == "注销账户":
            self.destroy_account()
        elif title == "开机自启动":
            self.toggle_auto_start_from_card()
        elif title == "立即锁屏":
            self.manual_lock()

    def manual_lock(self):
        result = ConfirmDialog(
            self,
            title="锁定系统",
            message="确定要锁定系统吗？\n锁定后将通过人脸识别解锁。",
            icon_type="info"
        )
        if result.exec_() == QDialog.Accepted:
            self.trigger_lock_screen()

    def change_password(self):
        dlg = InputDialog(
            self,
            title="修改密码",
            label1="旧密码",
            label2="新密码",
            placeholder1="请输入当前密码",
            placeholder2="请输入新密码",
            is_password=True
        )
        if dlg.exec_() == QDialog.Accepted:
            old_pwd, new_pwd = dlg.get_inputs()
            if not old_pwd or not new_pwd:
                QMessageBox.warning(self, "提示", "密码不能为空")
                return
            if len(new_pwd) < 6:
                QMessageBox.warning(self, "提示", "新密码长度不能少于6位")
                return
            pkg = package()
            if pkg.if_info_exist(self.username, old_pwd):
                if pkg.update_password(self.username, new_pwd):
                    pkg.insertdatalog(f"用户 {self.username} 修改了登录密码", self.username, None)
                    QMessageBox.information(self, "成功", "密码修改成功！")
                else:
                    QMessageBox.warning(self, "错误", "修改失败，请检查数据库连接")
            else:
                QMessageBox.warning(self, "提示", "旧密码错误")

    def view_logs(self):
        pkg = package()
        logs = pkg.getdatalog(self.username)
        if not logs:
            QMessageBox.information(self, "提示", "暂无操作日志")
            return
        from PyQt5.QtWidgets import (QTableWidget, QTableWidgetItem, QDialog, QVBoxLayout,
                                    QHBoxLayout, QPushButton, QLabel, QFrame)
        from PyQt5.QtGui import QColor
        from PyQt5.QtCore import Qt
        dialog = QDialog(self)
        dialog.setWindowTitle("操作日志")
        dialog.resize(900, 550)
        dialog.setMinimumSize(700, 400)
        dialog.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        dialog.setStyleSheet("""
            QDialog { background-color: #0d0d1a; }
            QLabel#titleLabel { color: #ffffff; font-size: 22px; font-weight: 300; }
            QTableWidget {
                background-color: #16162e; color: #e8edf5;
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-radius: 12px;
                gridline-color: rgba(255, 255, 255, 0.04);
                font-size: 13px;
                selection-background-color: rgba(74, 158, 255, 0.2);
                alternate-background-color: rgba(255, 255, 255, 0.02);
            }
            QTableWidget::item { padding: 10px 14px; }
            QHeaderView::section {
                background-color: #1a1a32; color: #8899bb;
                padding: 12px 14px; border: none;
                font-weight: 600; font-size: 13px;
            }
            QPushButton#refreshBtn {
                background-color: transparent;
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 8px; color: #aab8d0;
                font-size: 13px; padding: 8px 16px;
            }
            QPushButton#refreshBtn:hover {
                background-color: rgba(255, 255, 255, 0.04);
                border-color: rgba(255, 255, 255, 0.16);
                color: #ffffff;
            }
        """)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 25, 30, 25)
        main_layout.setSpacing(18)
        top_layout = QHBoxLayout()
        top_layout.setSpacing(0)
        left_layout = QHBoxLayout()
        left_layout.setSpacing(15)
        title_label = QLabel("📋 操作日志")
        title_label.setObjectName("titleLabel")
        left_layout.addWidget(title_label)
        count_badge = QFrame()
        count_badge.setStyleSheet("""
            QFrame {
                background-color: rgba(74, 158, 255, 0.15);
                border-radius: 12px; padding: 4px 14px;
            }
        """)
        count_layout = QHBoxLayout(count_badge)
        count_layout.setContentsMargins(10, 4, 10, 4)
        count_label = QLabel(f"共 {len(logs)} 条记录")
        count_label.setStyleSheet("color: #4a9eff; font-size: 13px; font-weight: 500;")
        count_layout.addWidget(count_label)
        left_layout.addWidget(count_badge)
        top_layout.addLayout(left_layout)
        top_layout.addStretch()
        refresh_btn = QPushButton("🔄 刷新")
        refresh_btn.setObjectName("refreshBtn")
        refresh_btn.setFixedHeight(34)
        top_layout.addWidget(refresh_btn)
        main_layout.addLayout(top_layout)
        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["时间", "用户", "操作记录"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setColumnWidth(0, 180)
        table.setColumnWidth(1, 120)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.setRowCount(len(logs))
        for i, log in enumerate(logs):
            table.setItem(i, 0, QTableWidgetItem(str(log['timestamp'])))
            table.setItem(i, 1, QTableWidgetItem(log['username']))
            table.setItem(i, 2, QTableWidgetItem(log['datalog']))
            item = table.item(i, 2)
            text = log['datalog']
            if '登录' in text or '解锁' in text:
                item.setForeground(QColor(0, 255, 136))
            elif '注销' in text or '退出' in text:
                item.setForeground(QColor(255, 71, 87))
            elif '注册' in text:
                item.setForeground(QColor(74, 158, 255))
            elif '修改' in text or '更换' in text:
                item.setForeground(QColor(255, 165, 2))
            else:
                item.setForeground(QColor(255, 255, 255))
        main_layout.addWidget(table)
        dialog.setLayout(main_layout)

        def refresh_logs():
            new_logs = package().getdatalog(self.username)
            if new_logs:
                table.setRowCount(len(new_logs))
                for i, log in enumerate(new_logs):
                    table.setItem(i, 0, QTableWidgetItem(str(log['timestamp'])))
                    table.setItem(i, 1, QTableWidgetItem(log['username']))
                    table.setItem(i, 2, QTableWidgetItem(log['datalog']))
                count_label.setText(f"共 {len(new_logs)} 条记录")

        refresh_btn.clicked.connect(refresh_logs)
        dialog.exec_()

    def enroll_face(self):
        result = ConfirmDialog(
            self,
            title="录入人脸",
            message="即将开始人脸采集，请正对摄像头并完成活体检测。",
            icon_type="info"
        )
        if result.exec_() != QDialog.Accepted:
            return
        QMessageBox.information(self, "提示", "请正对摄像头，按提示完成动作")
        detector = auto_hand_controller()
        success, faceinfo = detector.run(False)
        if not success or faceinfo is None:
            QMessageBox.warning(self, "提示", "人脸采集失败，请重试")
            return
        try:
            pkg = package()
            sql = "UPDATE userinfo SET userfaceinfo = %s WHERE username = %s"
            face_json = serialize_feature(faceinfo)
            pkg.cursor.execute(sql, (face_json, self.username))
            pkg.conn.commit()
            pkg.insertdatalog(f"用户 {self.username} 录入了人脸信息", self.username, None)
            QMessageBox.information(self, "成功", "人脸信息录入成功！")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存失败: {e}")

    def update_face(self):
        result = ConfirmDialog(
            self,
            title="更换人脸",
            message="确定要更换当前账户的人脸信息吗？\n原有数据将被覆盖。",
            icon_type="warning"
        )
        if result.exec_() != QDialog.Accepted:
            return
        QMessageBox.information(self, "提示", "请正对摄像头，按提示完成动作")
        detector = auto_hand_controller()
        success, faceinfo = detector.run(False)
        if not success or faceinfo is None:
            QMessageBox.warning(self, "提示", "人脸采集失败，请重试")
            return
        try:
            pkg = package()
            sql = "UPDATE userinfo SET userfaceinfo = %s WHERE username = %s"
            face_json = serialize_feature(faceinfo)
            pkg.cursor.execute(sql, (face_json, self.username))
            pkg.conn.commit()
            pkg.insertdatalog(f"用户 {self.username} 更换了人脸信息", self.username, None)
            QMessageBox.information(self, "成功", "人脸信息更换成功！")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存失败: {e}")

    def switch_account(self):
        result = ConfirmDialog(
            self,
            title="切换账户",
            message="确定要切换账户吗？\n当前会话将退出，返回登录界面。",
            icon_type="info"
        )
        if result.exec_() == QDialog.Accepted:
            pkg = package()
            pkg.insertdatalog(f"用户 {self.username} 切换账户", self.username, None)
            pkg.clear_current_user()
            self.close()
            self.login_win = loginwindow()
            self.login_win.show()

    def destroy_account(self):
        result = ConfirmDialog(
            self,
            title="注销账户",
            message=f"⚠️ 确定要永久注销账户「{self.username}」吗？\n此操作将删除所有数据且不可恢复！",
            icon_type="danger"
        )
        if result.exec_() != QDialog.Accepted:
            return
        try:
            pkg = package()
            pkg.insertdatalog(f"用户 {self.username} 注销了账户", self.username, None)
            pkg.clear_current_user()
            sql = "DELETE FROM userinfo WHERE username = %s"
            pkg.cursor.execute(sql, (self.username,))
            pkg.conn.commit()
            QMessageBox.information(self, "成功", f"账户「{self.username}」已成功注销")
            self.close()
            self.login_win = loginwindow()
            self.login_win.show()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"注销失败: {e}")

    def closeEvent(self, event):
        if self.tray_icon and self.tray_icon.isVisible():
            self.hide()
            self.tray_icon.showMessage(
                "Security System",
                f"程序已最小化到系统托盘",
                QSystemTrayIcon.Information,
                2000
            )
            event.ignore()
        else:
            pkg = package()
            pkg.insertdatalog(f"用户 {self.username} 退出系统", self.username, None)
            pkg.clear_current_user()
            if self.idle_timer:
                self.idle_timer.stop()
            event.accept()

    def get_idle_duration(self):
        try:
            class LASTINPUTINFO(ctypes.Structure):
                _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]
            last_input_info = LASTINPUTINFO()
            last_input_info.cbSize = ctypes.sizeof(LASTINPUTINFO)
            if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(last_input_info)):
                millis = ctypes.windll.kernel32.GetTickCount() - last_input_info.dwTime
                return millis / 1000.0
            return 0
        except:
            return 0

    def is_any_fullscreen_app_running(self):
        try:
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            if hwnd == 0:
                return False
            rect = wintypes.RECT()
            ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
            screen_width = ctypes.windll.user32.GetSystemMetrics(0)
            screen_height = ctypes.windll.user32.GetSystemMetrics(1)
            if rect.left <= 0 and rect.top <= 0 and rect.right >= screen_width and rect.bottom >= screen_height:
                return True
            return False
        except:
            return False

    def check_idle_and_scan(self):
        if self.is_any_fullscreen_app_running():
            self.idle_check_pending = False
            return
        idle_seconds = self.get_idle_duration()
        if idle_seconds >= 1800:
            if not self.idle_check_pending:
                self.idle_check_pending = True
                QTimer.singleShot(500, self.perform_safety_check)
        else:
            self.idle_check_pending = False

    def perform_safety_check(self):
        if self.get_idle_duration() < 1700:
            self.idle_check_pending = False
            return
        from .camera import camera_process
        cam = camera_process()
        if not cam.start():
            self.idle_check_pending = False
            return
        frame = None
        for _ in range(10):
            frame = cam.getframe()
            if frame is not None:
                break
            time.sleep(0.1)
        cam.release()
        cv2.destroyAllWindows()
        if frame is None:
            self.idle_check_pending = False
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_tensor = mtcnn(rgb)
        if face_tensor is None:
            self.idle_check_pending = False
            self.trigger_lock_screen()
            return
        face_tensor = face_tensor.to(device)
        with torch.no_grad():
            feature = resnet(face_tensor.unsqueeze(0))[0].cpu().numpy()
        pkg = package()
        current_user = pkg.get_current_user()
        if current_user is None:
            current_user = self.username
        try:
            sql = "SELECT userfaceinfo FROM userinfo WHERE username = %s"
            pkg.cursor.execute(sql, (current_user,))
            result = pkg.cursor.fetchone()
            if result and result['userfaceinfo']:
                stored_feat = decode_feature(result['userfaceinfo'])
                if stored_feat is None:
                    raise ValueError("库里的特征格式无法解析")
                sim = np.dot(feature, stored_feat) / (
                    np.linalg.norm(feature) * np.linalg.norm(stored_feat) + 1e-6
                )
                if sim > 0.6:
                    self.idle_check_pending = False
                    pkg.insertdatalog(f"空闲检测：用户 {current_user} 验证通过", current_user)
                    return
        except Exception as e:
            pass
        self.idle_check_pending = False
        self.trigger_lock_screen()

    def lock_workstation(self):
        """Windows API 系统锁屏。

        自定义锁屏负责刷脸解锁；这一层是系统级兜底——自定义窗口建不出来、
        或用户从托盘主动要求时，直接把会话交给 Windows 安全桌面。
        """
        try:
            pkg = package()
            pkg.insertdatalog(f"Windows API 锁屏：用户 {self.username}", self.username)
        except Exception as e:
            print(f"[锁屏] 审计写入失败: {e}")
        if not lock_workstation_api():
            print("[锁屏] LockWorkStation 调用失败")
            QMessageBox.warning(self, "锁屏失败", "Windows API 拒绝了锁屏请求")
            return False
        return True

    def trigger_lock_screen(self):
        try:
            pkg = package()
            pkg.insertdatalog(f"触发锁屏：用户 {self.username}", self.username)
            if self.lock_screen_active:
                return
            self.lock_screen_active = True
            self.hide()
            from .auto_ui import Lockscreen
            self.lock_screen_window = Lockscreen(main_window=self)
            self.lock_screen_window.showFullScreen()
        except Exception as e:
            print(f"触发锁屏失败: {e}")
            self.lock_screen_active = False
            self.show()
            self.lock_workstation()

    def start_quick_lock_check(self):
        if self.quick_lock_triggered:
            return
        idle_seconds = self.get_idle_duration()
        if idle_seconds >= 3600:
            self.quick_lock_triggered = True
            print("[快速锁屏] 检测到空闲3600秒，立即锁屏")
            self.trigger_lock_screen()
        else:
            remaining = 3600 - idle_seconds
            self.quick_lock_timer.start(int(remaining * 1000) + 100)

    def quick_lock_check(self):
        if self.quick_lock_triggered:
            return
        idle_seconds = self.get_idle_duration()
        if idle_seconds >= 3600:
            self.quick_lock_triggered = True
            print("[快速锁屏] 空闲3600秒，执行锁屏")
            self.trigger_lock_screen()
        else:
            remaining = 3600 - idle_seconds
            self.quick_lock_timer.start(int(remaining * 1000) + 100)


UN = ""