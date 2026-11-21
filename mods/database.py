"""数据访问层：连接、用户表、审计日志、在线状态。

安全约定（原来这三处是硬伤）：
  1. 连接口令只读环境变量 / 本地 .env（config.py），源码里不再有明文；
  2. password 列存 PBKDF2 哈希，历史明文在登录成功后自动升级；
  3. 所有 SQL 都走参数化查询，UI 层不再直接拼 UPDATE。
"""

from __future__ import annotations

import socket

import pymysql
from pymysql import MySQLError
from PyQt5 import QtWidgets

from .config import db_config
from .security import hash_password, verify_password, is_hashed

PASSWORD_COLUMN_WIDTH = 255
_WIDEN_SQL = {
    "password": "ALTER TABLE userinfo MODIFY COLUMN password VARCHAR(255) NOT NULL",
    "userwindowpassword": "ALTER TABLE userinfo MODIFY COLUMN userwindowpassword VARCHAR(255)",
}


class package():
    def __init__(self, parent=None, **overrides):
        self.parent = parent
        self.conn = None
        self.cursor = None
        try:
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
            self.creatdbtable()
            self.create_current_user()
            self.createdatalog()
        except (MySQLError, RuntimeError) as e:
            self.conn = None
            self.cursor = None
            self._warn(f"数据库不可用：{e}")

    def _ready(self):
        return self.conn is not None and self.cursor is not None

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
            self._ensure_password_width()
        except MySQLError as e:
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
            self.cursor.execute(
                """INSERT INTO userinfo(username, password, userfaceinfo, userwindowpassword)
                   VALUES(%s, %s, %s, %s)""",
                (username, hash_password(userpassword), userfaceinfo, windowpassword),
            )
            self.conn.commit()
            return True
        except (MySQLError, ValueError) as e:
            self._warn(f"注册写入失败: {e}")
            return False

    def if_username_exist(self, username):
        if not self._ready():
            return False
        try:
            self.cursor.execute("SELECT 1 FROM userinfo WHERE username = %s", (username,))
            return self.cursor.fetchone() is not None
        except MySQLError as e:
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
            self.cursor.execute(
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
        except MySQLError as e:
            self._warn(f"登录校验失败: {e}")
            return False

    def _upgrade_password(self, username, plain_password):
        try:
            self.cursor.execute(
                "UPDATE userinfo SET password = %s WHERE username = %s",
                (hash_password(plain_password), username),
            )
            self.conn.commit()
            print(f"[DB] 用户 {username} 的明文口令已自动升级为哈希")
        except MySQLError as e:
            print(f"[DB] 口令升级失败，下次登录会重试: {e}")

    def update_password(self, username, new_password):
        """改密统一走数据层，UI 不再自己写 SQL。"""
        if not self._ready():
            return False
        try:
            self.cursor.execute(
                "UPDATE userinfo SET password = %s WHERE username = %s",
                (hash_password(new_password), username),
            )
            self.conn.commit()
            return True
        except (MySQLError, ValueError) as e:
            self._warn(f"修改密码失败: {e}")
            return False

    def insertdatalog(self, datalog, username, timestamp=None):
        if not self._ready():
            return False
        try:
            if timestamp:
                self.cursor.execute(
                    "INSERT INTO userdatalog(username, datalog, timestamp) VALUES(%s,%s,%s)",
                    (username, datalog, timestamp),
                )
            else:
                self.cursor.execute(
                    "INSERT INTO userdatalog(username, datalog) VALUES(%s,%s)",
                    (username, datalog),
                )
            self.conn.commit()
            return True
        except MySQLError as e:
            self._warn(f"写入日志失败: {e}")
            return False

    def createdatalog(self):
        if not self._ready():
            return False
        try:
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS userdatalog(
                    username VARCHAR(50) NOT NULL,
                    datalog TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            ''')
            self.conn.commit()
            return True
        except MySQLError as e:
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
            self.cursor.execute(sql, tuple(params))
            return self.cursor.fetchall()
        except MySQLError as e:
            self._warn(f"获取日志表失败: {e}")
            return []

    def create_current_user(self):
        if not self._ready():
            return False
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
        except MySQLError as e:
            self._warn(f"数据库当前用户表创建失败: {e}")
            return False

    def insertinfo_to_current_user(self, device_id, username, status, login_time, updata_time):
        """同一账号在别处登录时被挤下线；device_id 以本机主机名为准。"""
        if not self._ready():
            return False
        device_id = socket.gethostname()
        try:
            self.cursor.execute(
                "UPDATE user_current SET status = 'offline' WHERE username = %s AND device_id != %s",
                (username, device_id),
            )
            self.cursor.execute(
                """INSERT INTO user_current (device_id, username, status)
                   VALUES (%s, %s, 'online')
                   ON DUPLICATE KEY UPDATE status = 'online'""",
                (device_id, username),
            )
            self.conn.commit()
            return True
        except MySQLError as e:
            self._warn(f"设置当前用户失败: {e}")
            return False

    def get_current_user(self):
        if not self._ready():
            return None
        device_id = socket.gethostname()
        try:
            self.cursor.execute(
                "SELECT username FROM user_current WHERE device_id = %s AND status = 'online'",
                (device_id,),
            )
            result = self.cursor.fetchone()
            return result['username'] if result else None
        except MySQLError as e:
            self._warn(f"获取当前用户失败: {e}")
            return None

    def clear_current_user(self):
        if not self._ready():
            return False
        device_id = socket.gethostname()
        try:
            self.cursor.execute(
                "UPDATE user_current SET status = 'offline' WHERE device_id = %s",
                (device_id,),
            )
            self.conn.commit()
            return True
        except MySQLError as e:
            self._warn(f"清除当前用户失败: {e}")
            return False

    def is_empty(self):
        if not self._ready():
            return True
        try:
            self.cursor.execute("SELECT 1 FROM userinfo LIMIT 1")
            return self.cursor.fetchone() is None
        except MySQLError as e:
            self._warn(f"检查用户表失败: {e}")
            return True
