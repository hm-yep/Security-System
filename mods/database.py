"""数据访问层：连接、用户表、审计日志、在线状态。

安全约定（原来这三处是硬伤）：
  1. 连接口令只读环境变量 / 本地 .env（config.py），源码里不再有明文；
  2. password 列存 PBKDF2 哈希，历史明文在登录成功后自动升级；
  3. 所有 SQL 都走参数化查询，UI 层不再直接拼 UPDATE。

部署形态：
  - 源码运行：默认 MySQL（SECURITY_DB_BACKEND=mysql），凭据来自 .env；
  - 打包分发：默认 SQLite（SECURITY_DB_BACKEND=sqlite），库文件放在
    %APPDATA%\\SecuritySystem\\security.db，用户下载解压即用，无需装 MySQL。
"""

from __future__ import annotations

import os
import sqlite3
import socket
import sys

import pymysql
from pymysql import MySQLError
from PyQt5 import QtWidgets

from .config import db_config, load_dotenv
from .security import hash_password, verify_password, is_hashed

PASSWORD_COLUMN_WIDTH = 255
_DB_ERRORS = (MySQLError, sqlite3.Error)
_WIDEN_SQL = {
    "password": "ALTER TABLE userinfo MODIFY COLUMN password VARCHAR(255) NOT NULL",
    "userwindowpassword": "ALTER TABLE userinfo MODIFY COLUMN userwindowpassword VARCHAR(255)",
}


def resolve_backend(explicit=None) -> str:
    """决定用哪种后端：显式参数 > 环境变量 > 打包版 sqlite / 源码 mysql。"""
    if explicit in ("mysql", "sqlite"):
        return explicit
    load_dotenv()
    env = os.getenv("SECURITY_DB_BACKEND", "").strip().lower()
    if env in ("mysql", "sqlite"):
        return env
    return "sqlite" if getattr(sys, "frozen", False) else "mysql"


def default_sqlite_path() -> str:
    """用户数据目录，避免把库写进 Program Files 这类只读位置。"""
    base = os.getenv("APPDATA") or os.path.expanduser("~")
    return os.path.join(base, "SecuritySystem", "security.db")


class package():
    def __init__(self, parent=None, **overrides):
        self.parent = parent
        self.conn = None
        self.cursor = None
        self.backend = resolve_backend(overrides.pop("backend", None))
        try:
            if self.backend == "sqlite":
                self._connect_sqlite(overrides.get("sqlite_path"))
            else:
                self._connect_mysql(overrides)
            self.creatdbtable()
            self.create_current_user()
            self.createdatalog()
        except _DB_ERRORS + (RuntimeError, OSError) as e:
            self.conn = None
            self.cursor = None
            self._warn(f"数据库不可用：{e}")

    def _connect_mysql(self, overrides):
        config = db_config()
        config.update(overrides)
        self.conn = pymysql.connect(
            host=config["host"],
            user=config["user"],
            password=config["password"],
            database=config["database"],
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
        )
        self.cursor = self.conn.cursor()

    def _connect_sqlite(self, path=None):
        path = path or os.getenv("SECURITY_DB_PATH") or default_sqlite_path()
        folder = os.path.dirname(os.path.abspath(path))
        if folder:
            os.makedirs(folder, exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=5, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # 检测线程写审计、UI 线程读写用户表，WAL 让读写互不阻塞。
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.cursor = self.conn.cursor()
        print(f"[DB] SQLite 后端: {path}")

    def _exec(self, sql, params=None):
        """统一占位符：业务代码继续写 %s，SQLite 分支自动换成 ?。"""
        if self.backend == "sqlite":
            sql = sql.replace("%s", "?")
        if params is None:
            return self.cursor.execute(sql)
        return self.cursor.execute(sql, params)

    def _ready(self):
        return self.conn is not None and self.cursor is not None

    @property
    def ready(self) -> bool:
        """对外暴露连接可用性，避免调用方依赖内部方法名。"""
        return self._ready()

    def _warn(self, message):
        print(f"[DB] {message}")
        if self.parent is not None:
            QtWidgets.QMessageBox.information(self.parent, "连接超时", message)

    def close(self):
        """显式释放连接；调用方在长生命周期里应复用同一个 package 实例。"""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            self.cursor = None

    def creatdbtable(self):
        if not self._ready():
            return
        if self.backend == "sqlite":
            sql = """
            CREATE TABLE IF NOT EXISTS userinfo(
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                userfaceinfo TEXT,
                userwindowpassword TEXT
            );
            """
        else:
            sql = """
            CREATE TABLE IF NOT EXISTS userinfo(
                username VARCHAR(50) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                userfaceinfo TEXT,
                userwindowpassword VARCHAR(255)
            )ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        try:
            self.cursor.execute(sql)
            self.conn.commit()
            if self.backend != "sqlite":
                self._ensure_password_width()
        except _DB_ERRORS as e:
            self._warn(f"创建用户表失败: {e}")

    def _ensure_password_width(self):
        """旧库的 password 是 VARCHAR(20)，装不下哈希串；按需做一次幂等迁移。"""
        for column, alter in _WIDEN_SQL.items():
            try:
                self.cursor.execute(
                    """SELECT CHARACTER_MAXIMUM_LENGTH AS width
                       FROM information_schema.COLUMNS
                       WHERE TABLE_SCHEMA = DATABASE()
                         AND TABLE_NAME = 'userinfo' AND COLUMN_NAME = %s""",
                    (column,),
                )
                row = self.cursor.fetchone()
                width = int(row["width"]) if row and row.get("width") is not None else 0
                if 0 < width < PASSWORD_COLUMN_WIDTH:
                    self.cursor.execute(alter)
                    self.conn.commit()
                    print(f"[DB] 已把 {column} 扩到 VARCHAR(255) 以容纳口令哈希")
            except Exception as e:
                print(f"[DB] 列宽迁移跳过（{column}）: {e}")

    def getinfo(self, username, userpassword, userfaceinfo, windowpassword):
        """注册：口令一律哈希后再入库。"""
        if not self._ready():
            return False
        try:
            self._exec(
                """INSERT INTO userinfo(username, password, userfaceinfo, userwindowpassword)
                   VALUES(%s, %s, %s, %s)""",
                (username, hash_password(userpassword), userfaceinfo, windowpassword),
            )
            self.conn.commit()
            return True
        except _DB_ERRORS + (ValueError,) as e:
            self._warn(f"注册写入失败: {e}")
            return False

    def get_face_feature(self, username):
        """读取人脸特征 JSON；未录入或数据库不可用返回 None。"""
        if not self._ready():
            return None
        try:
            self._exec(
                "SELECT userfaceinfo FROM userinfo WHERE username = %s",
                (username,),
            )
            row = self.cursor.fetchone()
            return row["userfaceinfo"] if row else None
        except _DB_ERRORS as e:
            self._warn(f"读取人脸特征失败: {e}")
            return None

    def update_face_feature(self, username, feature_json):
        if not self._ready():
            return False
        try:
            self._exec(
                "UPDATE userinfo SET userfaceinfo = %s WHERE username = %s",
                (feature_json, username),
            )
            self.conn.commit()
            return True
        except _DB_ERRORS as e:
            self._warn(f"更新人脸特征失败: {e}")
            return False

    def delete_user(self, username):
        """注销账户；审计写入与在线状态清理由调用方负责。"""
        if not self._ready():
            return False
        try:
            self._exec("DELETE FROM userinfo WHERE username = %s", (username,))
            self.conn.commit()
            return True
        except _DB_ERRORS as e:
            self._warn(f"注销账户失败: {e}")
            return False

    def if_username_exist(self, username):
        if not self._ready():
            return False
        try:
            self._exec("SELECT 1 FROM userinfo WHERE username = %s", (username,))
            return self.cursor.fetchone() is not None
        except _DB_ERRORS as e:
            self._warn(f"查询用户失败: {e}")
            return False

    def if_info_exist(self, username, userpassword):
        """登录校验：兼容历史明文，并在通过后把该行升级成哈希。

        老库直接切哈希会让所有存量用户登不进来，所以这里做的是渐进迁移，
        不需要停服刷数据。
        """
        if not self._ready():
            return False
        try:
            self._exec(
                "SELECT password FROM userinfo WHERE username = %s", (username,)
            )
            row = self.cursor.fetchone()
            if row is None:
                return False
            stored = row["password"]
            if not verify_password(userpassword, stored):
                return False
            if not is_hashed(stored):
                self._upgrade_password(username, userpassword)
            return True
        except _DB_ERRORS as e:
            self._warn(f"登录校验失败: {e}")
            return False

    def _upgrade_password(self, username, plain_password):
        try:
            self._exec(
                "UPDATE userinfo SET password = %s WHERE username = %s",
                (hash_password(plain_password), username),
            )
            self.conn.commit()
            print(f"[DB] 用户 {username} 的明文口令已自动升级为哈希")
        except _DB_ERRORS as e:
            print(f"[DB] 口令升级失败，下次登录会重试: {e}")

    def update_password(self, username, new_password):
        """改密统一走数据层，UI 不再自己写 SQL。"""
        if not self._ready():
            return False
        try:
            self._exec(
                "UPDATE userinfo SET password = %s WHERE username = %s",
                (hash_password(new_password), username),
            )
            self.conn.commit()
            return True
        except _DB_ERRORS + (ValueError,) as e:
            self._warn(f"修改密码失败: {e}")
            return False

    def insertdatalog(self, datalog, username, timestamp=None):
        if not self._ready():
            return False
        try:
            if timestamp:
                self._exec(
                    "INSERT INTO userdatalog(username, datalog, timestamp) VALUES(%s,%s,%s)",
                    (username, datalog, timestamp),
                )
            else:
                self._exec(
                    "INSERT INTO userdatalog(username, datalog) VALUES(%s,%s)",
                    (username, datalog),
                )
            self.conn.commit()
            return True
        except _DB_ERRORS as e:
            self._warn(f"写入日志失败: {e}")
            return False

    def createdatalog(self):
        if not self._ready():
            return False
        suffix = "" if self.backend == "sqlite" else " ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
        try:
            self.cursor.execute(
                f'''
                CREATE TABLE IF NOT EXISTS userdatalog(
                    username VARCHAR(50) NOT NULL,
                    datalog TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                ){suffix};
                '''
            )
            self.conn.commit()
            return True
        except _DB_ERRORS as e:
            self._warn(f"创建日志表失败: {e}")
            return False

    def getdatalog(self, username):
        if not self._ready():
            return []
        try:
            sql = "SELECT username, datalog, timestamp FROM userdatalog"
            params = []
            if username:
                sql += " WHERE username = %s"
                params.append(username)
            sql += " ORDER BY timestamp DESC"
            self._exec(sql, tuple(params))
            return self.cursor.fetchall()
        except _DB_ERRORS as e:
            self._warn(f"获取日志表失败: {e}")
            return []

    def count_today_logs(self, keywords, username=None):
        """统计今天包含任一关键词的审计条数，供主界面安全看板使用。

        关键词匹配放在 SQL 里做（同样走参数化），避免把整张日志表拉回内存；
        数据库不可用时返回 None，由界面显示占位符。
        """
        if not self._ready() or not keywords:
            return None
        try:
            clauses = " OR ".join("datalog LIKE %s" for _ in keywords)
            today = (
                "date(timestamp) = date('now','localtime')"
                if self.backend == "sqlite"
                else "DATE(timestamp) = CURDATE()"
            )
            sql = (
                "SELECT COUNT(*) AS total FROM userdatalog "
                f"WHERE {today} AND ({clauses})"
            )
            params = [f"%{word}%" for word in keywords]
            if username:
                sql += " AND username = %s"
                params.append(username)
            self._exec(sql, tuple(params))
            row = self.cursor.fetchone()
            return int(row["total"]) if row else 0
        except _DB_ERRORS as e:
            self._warn(f"统计审计日志失败: {e}")
            return None

    def create_current_user(self):
        if not self._ready():
            return False
        if self.backend == "sqlite":
            sql = '''
            CREATE TABLE IF NOT EXISTS user_current (
                device_id VARCHAR(100) NOT NULL,
                username VARCHAR(50) NOT NULL,
                status VARCHAR(20) DEFAULT 'offline',
                login_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                update_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (device_id, username)
            );
            '''
        else:
            sql = '''
            CREATE TABLE IF NOT EXISTS user_current (
                device_id VARCHAR(100) NOT NULL,
                username VARCHAR(50) NOT NULL,
                status VARCHAR(20) DEFAULT 'offline',
                login_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (device_id, username)
            )ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            '''
        try:
            self.cursor.execute(sql)
            self.conn.commit()
            return True
        except _DB_ERRORS as e:
            self._warn(f"数据库当前用户表创建失败: {e}")
            return False

    def insertinfo_to_current_user(self, device_id, username, status, login_time, updata_time):
        """同一账号在别处登录时被挤下线；device_id 以本机主机名为准。"""
        if not self._ready():
            return False
        device_id = socket.gethostname()
        try:
            self._exec(
                "UPDATE user_current SET status = 'offline' WHERE username = %s AND device_id != %s",
                (username, device_id),
            )
            if self.backend == "sqlite":
                sql = """INSERT INTO user_current (device_id, username, status)
                         VALUES (%s, %s, 'online')
                         ON CONFLICT(device_id, username) DO UPDATE SET status = 'online'"""
            else:
                sql = """INSERT INTO user_current (device_id, username, status)
                         VALUES (%s, %s, 'online')
                         ON DUPLICATE KEY UPDATE status = 'online'"""
            self._exec(sql, (device_id, username))
            self.conn.commit()
            return True
        except _DB_ERRORS as e:
            self._warn(f"设置当前用户失败: {e}")
            return False

    def get_current_user(self):
        if not self._ready():
            return None
        device_id = socket.gethostname()
        try:
            self._exec(
                "SELECT username FROM user_current WHERE device_id = %s AND status = 'online'",
                (device_id,),
            )
            result = self.cursor.fetchone()
            return result['username'] if result else None
        except _DB_ERRORS as e:
            self._warn(f"获取当前用户失败: {e}")
            return None

    def clear_current_user(self):
        if not self._ready():
            return False
        device_id = socket.gethostname()
        try:
            self._exec(
                "UPDATE user_current SET status = 'offline' WHERE device_id = %s",
                (device_id,),
            )
            self.conn.commit()
            return True
        except _DB_ERRORS as e:
            self._warn(f"清除当前用户失败: {e}")
            return False

    def is_empty(self):
        if not self._ready():
            return True
        try:
            self._exec("SELECT 1 FROM userinfo LIMIT 1")
            return self.cursor.fetchone() is None
        except _DB_ERRORS as e:
            self._warn(f"检查用户表失败: {e}")
            return True
