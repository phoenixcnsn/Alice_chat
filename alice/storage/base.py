"""
SQLite 存储基类 — StyleStore 和 MemoryStore 共用
"""
import sqlite3
from pathlib import Path


class SQLiteStore:
    """所有 SQLite 存储的基类，提供统一的连接管理"""

    def __init__(self, db_path: str):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        """创建并返回一个配置好的 SQLite 连接"""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn
