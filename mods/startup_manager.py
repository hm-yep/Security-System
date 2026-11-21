# startup_manager.py
import sys
import os
import winreg

def get_app_path_with_args():
    """获取当前程序路径并带上 --auto 参数"""
    if getattr(sys, 'frozen', False):
        # 打包成exe后
        return f'"{sys.executable}" --auto'
    else:
        # 开发环境：使用python解释器运行主脚本
        main_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.py')
        return f'"{sys.executable}" "{main_script}" --auto'

def enable_startup():
    """开启开机自启动（写入当前用户注册表）"""
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE)
        winreg.SetValueEx(key, "SecuritySystem", 0, winreg.REG_SZ, get_app_path_with_args())
        winreg.CloseKey(key)
        return True
    except Exception as e:
        print(f"开启自启动失败: {e}")
        return False

def disable_startup():
    """关闭开机自启动（删除注册表项）"""
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE)
        winreg.DeleteValue(key, "SecuritySystem")
        winreg.CloseKey(key)
        return True
    except FileNotFoundError:
        return True
    except Exception as e:
        print(f"关闭自启动失败: {e}")
        return False

def is_startup_enabled():
    """检查是否已开启开机自启动"""
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_READ)
        try:
            winreg.QueryValueEx(key, "SecuritySystem")
            winreg.CloseKey(key)
            return True
        except FileNotFoundError:
            winreg.CloseKey(key)
            return False
    except Exception:
        return False