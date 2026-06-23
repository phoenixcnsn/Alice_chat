"""
SQLite 记忆存储层 — 全量留存 + FTS5 全文检索 + 滚动摘要

三层存储:
  1. messages       — 原始消息全量留存，不可物理删除（底层底片）
  2. summaries      — 每 50 条消息生成一次滚动摘要
  3. memory_fragments — 提取的事实/偏好/情绪碎片，带 FTS5 全文索引

敏感字段加密: 消息内容使用 Fernet 对称加密存储
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ------------------------------------------------------------
# 加密工具
# ------------------------------------------------------------
try:
    from cryptography.fernet import Fernet
    _HAS_CRYPTO = True
except ImportError:
    _HAS_CRYPTO = False


class Crypto:
    """Fernet 对称加密"""

    def __init__(self, key: Optional[bytes] = None):
        if not _HAS_CRYPTO:
            raise ImportError("需要 cryptography: pip install cryptography")
        if key is None:
            key = Fernet.generate_key()
        self._fernet = Fernet(key)
        self.key = key

    def encrypt(self, plain: str) -> bytes:
        return self._fernet.encrypt(plain.encode("utf-8"))

    def decrypt(self, cipher: bytes) -> str:
        return self._fernet.decrypt(cipher).decode("utf-8")


# ------------------------------------------------------------
# 数据模型
# ------------------------------------------------------------
@dataclass
class Message:
    id: int = 0
    role: str = ""
    content: str = ""
    preset_name: str = ""
    created_at: str = ""
    metadata: Dict = field(default_factory=dict)

@dataclass
class MemoryFragment:
    id: int = 0
    fragment_type: str = ""     # fact / preference / emotion / event
    content: str = ""
    source_msg_ids: List[int] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    importance: float = 0.5
    created_at: str = ""
    updated_at: str = ""


# ------------------------------------------------------------
# SQLite 存储
# ------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,              -- 消息内容
    preset_name TEXT DEFAULT '',
    created_at REAL NOT NULL,
    metadata TEXT DEFAULT '{}'          -- JSON
);

CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_msg_id INTEGER NOT NULL,
    end_msg_id INTEGER NOT NULL,
    summary BLOB NOT NULL,              -- 加密存储
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_fragments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fragment_type TEXT NOT NULL CHECK(fragment_type IN ('fact','preference','emotion','event')),
    content TEXT NOT NULL,              -- 消息内容
    source_msg_ids TEXT DEFAULT '[]',
    entities TEXT DEFAULT '[]',
    importance REAL DEFAULT 0.5,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_msg_created ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_msg_preset ON messages(preset_name);
CREATE INDEX IF NOT EXISTS idx_mem_type ON memory_fragments(fragment_type);
CREATE INDEX IF NOT EXISTS idx_mem_importance ON memory_fragments(importance);
"""


from alice.storage.base import SQLiteStore


class MemoryStore(SQLiteStore):
    """SQLite 记忆存储"""

    def __init__(self, db_path: str = "data/memory.db", crypto_key: Optional[bytes] = None):
        super().__init__(db_path)
        self.crypto = Crypto(crypto_key) if _HAS_CRYPTO else None
        self._init_db()

    def _init_db(self):
        conn = self._connect()
        conn.executescript(SCHEMA)
        conn.commit()
        conn.close()

    def _encrypt(self, text: str) -> str:
        """加密文本。
        - 无 crypto_key: 明文存储，以支持 FTS5/LIKE 搜索。
        - 有 crypto_key: Fernet 加密后 base64 编码存储。
          注意：加密后 LIKE/FTS5 搜索将不可用。
        """
        if self.crypto is None:
            return text
        encrypted_bytes = self.crypto.encrypt(text)
        return base64.b64encode(encrypted_bytes).decode("utf-8")

    def _decrypt(self, data: str) -> str:
        """解密文本。无 crypto_key 时原样返回。"""
        if self.crypto is None:
            return data
        try:
            encrypted_bytes = base64.b64decode(data.encode("utf-8"))
            return self.crypto.decrypt(encrypted_bytes)
        except Exception:
            # 兼容旧数据（可能是明文或损坏数据）
            return data

    # ----------------------------------------------------------------
    # 消息存储（全量留存，不可删除）
    # ----------------------------------------------------------------
    def save_message(self, role: str, content: str, preset_name: str = "",
                     metadata: Dict = None) -> int:
        """保存原始消息，返回消息 ID"""
        conn = self._connect()
        encrypted = self._encrypt(content)
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        conn.execute(
            "INSERT INTO messages (role, content, preset_name, created_at, metadata) VALUES (?,?,?,?,?)",
            (role, encrypted, preset_name, time.time(), meta_json)
        )
        conn.commit()
        msg_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()
        return msg_id

    def get_messages(self, limit: int = 100, offset: int = 0,
                     preset_name: str = "") -> List[Message]:
        """获取消息列表（最近优先）"""
        conn = self._connect()
        if preset_name:
            rows = conn.execute(
                "SELECT * FROM messages WHERE preset_name=? ORDER BY id DESC LIMIT ? OFFSET ?",
                (preset_name, limit, offset)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM messages ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset)
            ).fetchall()
        conn.close()
        return [Message(
            id=r["id"], role=r["role"],
            content=self._decrypt(r["content"]),
            preset_name=r["preset_name"], created_at=r["created_at"],
            metadata=json.loads(r["metadata"])
        ) for r in reversed(rows)]

    def get_message_count(self, preset_name: str = "") -> int:
        conn = self._connect()
        if preset_name:
            row = conn.execute("SELECT COUNT(*) FROM messages WHERE preset_name=?", (preset_name,)).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM messages").fetchone()
        conn.close()
        return row[0]

    def get_messages_since(self, since_id: int, preset_name: str = "") -> List[Message]:
        conn = self._connect()
        if preset_name:
            rows = conn.execute(
                "SELECT * FROM messages WHERE id > ? AND preset_name=? ORDER BY id",
                (since_id, preset_name)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM messages WHERE id > ? ORDER BY id", (since_id,)
            ).fetchall()
        conn.close()
        return [Message(
            id=r["id"], role=r["role"],
            content=self._decrypt(r["content"]),
            preset_name=r["preset_name"], created_at=r["created_at"],
            metadata=json.loads(r["metadata"])
        ) for r in rows]

    # ----------------------------------------------------------------
    # 滚动摘要
    # ----------------------------------------------------------------
    def get_last_summary_end_id(self) -> int:
        conn = self._connect()
        row = conn.execute("SELECT COALESCE(MAX(end_msg_id), 0) FROM summaries").fetchone()
        conn.close()
        return row[0]

    def save_summary(self, start_msg_id: int, end_msg_id: int, summary: str):
        conn = self._connect()
        conn.execute(
            "INSERT INTO summaries (start_msg_id, end_msg_id, summary, created_at) VALUES (?,?,?,?)",
            (start_msg_id, end_msg_id, self._encrypt(summary), time.time())
        )
        conn.commit()
        conn.close()

    def get_all_summaries(self) -> List[str]:
        conn = self._connect()
        rows = conn.execute("SELECT summary FROM summaries ORDER BY id").fetchall()
        conn.close()
        return [self._decrypt(r["summary"]) for r in rows]

    # ----------------------------------------------------------------
    # 记忆碎片
    # ----------------------------------------------------------------
    def save_fragment(self, fragment: MemoryFragment) -> int:
        conn = self._connect()
        now = time.time()
        conn.execute(
            """INSERT INTO memory_fragments
               (fragment_type, content, source_msg_ids, entities, importance, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?)""",
            (fragment.fragment_type, self._encrypt(fragment.content),
             json.dumps(fragment.source_msg_ids, ensure_ascii=False),
             json.dumps(fragment.entities, ensure_ascii=False),
             fragment.importance, now, now)
        )
        conn.commit()
        fid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()
        return fid

    def update_fragment(self, fragment_id: int, **kwargs):
        if not kwargs:
            return
        conn = self._connect()
        sets = []
        vals = []
        for k, v in kwargs.items():
            if k == "content":
                v = self._encrypt(v)
            elif k in ("source_msg_ids", "entities"):
                v = json.dumps(v, ensure_ascii=False)
            sets.append(f"{k}=?")
            vals.append(v)
        vals.append(time.time())
        vals.append(fragment_id)
        conn.execute(
            f"UPDATE memory_fragments SET {','.join(sets)}, updated_at=? WHERE id=?",
            vals
        )
        conn.commit()
        conn.close()

    def get_fragments_by_ids(self, ids: List[int]) -> List[MemoryFragment]:
        if not ids:
            return []
        conn = self._connect()
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT * FROM memory_fragments WHERE id IN ({placeholders}) ORDER BY importance DESC",
            ids
        ).fetchall()
        conn.close()
        return [MemoryFragment(
            id=r["id"], fragment_type=r["fragment_type"],
            content=self._decrypt(r["content"]),
            source_msg_ids=json.loads(r["source_msg_ids"]),
            entities=json.loads(r["entities"]),
            importance=r["importance"],
            created_at=r["created_at"], updated_at=r["updated_at"]
        ) for r in rows]

    # ----------------------------------------------------------------
    # 关键词检索（LIKE + 分词）
    # ----------------------------------------------------------------
    def search_fts(self, query: str, limit: int = 10) -> List[int]:
        """
        关键词检索，返回 memory_fragment IDs。
        使用 LIKE 匹配 content 字段。

        注意：启用加密后 LIKE 匹配不可用，返回空列表。
        """
        if self.crypto is not None:
            return []  # 加密存储时无法做 LIKE 搜索

        conn = self._connect()
        ids = []
        # 对每个关键词做 LIKE 匹配
        keywords = [k for k in query.replace(" ", "").split("，") if k]
        if not keywords:
            # 按单字拆分
            keywords = list(query[:10])

        for kw in keywords[:5]:
            try:
                rows = conn.execute(
                    "SELECT id, importance FROM memory_fragments WHERE "
                    "content LIKE ? ORDER BY importance DESC LIMIT ?",
                    (f"%{kw}%", max(1, limit // len(keywords)))
                ).fetchall()
                for r in rows:
                    if r[0] not in ids:
                        ids.append(r[0])
            except Exception:
                pass
        conn.close()
        return ids[:limit]

    def search_fragments(self, query: str, limit: int = 10) -> List[MemoryFragment]:
        """关键词检索记忆碎片。加密模式下返回空列表。"""
        if self.crypto is not None:
            return []
        ids = self.search_fts(query, limit)
        return self.get_fragments_by_ids(ids)

    def get_fragments_by_entities(self, entities: List[str], limit: int = 10) -> List[MemoryFragment]:
        """按实体聚合检索。加密模式下返回空列表。"""
        if not entities:
            return []
        if self.crypto is not None:
            return []  # 加密存储时无法做 LIKE 搜索
        conn = self._connect()
        # 用 LIKE 匹配 entities JSON 字段
        rows = []
        for entity in entities[:5]:
            r = conn.execute(
                "SELECT * FROM memory_fragments WHERE entities LIKE ? ORDER BY importance DESC LIMIT ?",
                (f"%{entity}%", max(1, limit // len(entities)))
            ).fetchall()
            rows.extend(r)
        conn.close()

        seen = set()
        result = []
        for r in sorted(rows, key=lambda x: x["importance"], reverse=True):
            if r["id"] not in seen:
                seen.add(r["id"])
                result.append(MemoryFragment(
                    id=r["id"], fragment_type=r["fragment_type"],
                    content=self._decrypt(r["content"]),
                    source_msg_ids=json.loads(r["source_msg_ids"]),
                    entities=json.loads(r["entities"]),
                    importance=r["importance"],
                    created_at=r["created_at"], updated_at=r["updated_at"]
                ))
        return result[:limit]

    def get_all_fragments_by_type(self, fragment_type: str) -> List[MemoryFragment]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM memory_fragments WHERE fragment_type=? ORDER BY importance DESC",
            (fragment_type,)
        ).fetchall()
        conn.close()
        return [MemoryFragment(
            id=r["id"], fragment_type=r["fragment_type"],
            content=self._decrypt(r["content"]),
            source_msg_ids=json.loads(r["source_msg_ids"]),
            entities=json.loads(r["entities"]),
            importance=r["importance"],
            created_at=r["created_at"], updated_at=r["updated_at"]
        ) for r in rows]

    # ----------------------------------------------------------------
    # 记忆衰减与强化（遗忘机制）
    # ----------------------------------------------------------------
    def reinforce_fragment(self, fragment_id: int, boost: float = 0.15):
        """
        强化记忆碎片：提升 importance 权重。

        公式: new_weight = min(1.0, old_weight * 0.95 + boost)

        Args:
            fragment_id: 碎片 ID
            boost: 单次强化增量 (0~1)，默认 0.15
        """
        conn = self._connect()
        row = conn.execute(
            "SELECT importance FROM memory_fragments WHERE id=?", (fragment_id,)
        ).fetchone()
        if row is None:
            conn.close()
            return
        old_weight = row["importance"]
        new_weight = min(1.0, old_weight * 0.95 + boost)
        now = time.time()
        conn.execute(
            "UPDATE memory_fragments SET importance=?, updated_at=? WHERE id=?",
            (new_weight, now, fragment_id),
        )
        conn.commit()
        conn.close()

    def decay_fragments(
        self,
        half_life_days: float = 30.0,
        min_weight: float = 0.1,
    ) -> int:
        """
        对所有记忆碎片应用指数衰减。

        公式: weight(t) = weight_0 * (1/2)^(t / halfLife)

        只更新上次更新距今超过 1 天的碎片（避免频繁更新刚操作的碎片）。

        Args:
            half_life_days: 半衰期（天），默认 30 天
            min_weight: 最低权重阈值

        Returns:
            被降权到 min_weight 以下的碎片数量
        """
        conn = self._connect()
        now = time.time()
        half_life_seconds = half_life_days * 86400.0
        decay_factor_per_second = 0.5 ** (1.0 / half_life_seconds)
        one_day_ago = now - 86400.0

        # 获取超过 1 天未更新的碎片
        rows = conn.execute(
            "SELECT id, importance, updated_at FROM memory_fragments WHERE updated_at < ?",
            (one_day_ago,)
        ).fetchall()

        below_threshold = 0
        for r in rows:
            age_seconds = now - r["updated_at"]
            decay_multiplier = decay_factor_per_second ** age_seconds
            new_weight = r["importance"] * decay_multiplier
            if new_weight < min_weight:
                below_threshold += 1
            conn.execute(
                "UPDATE memory_fragments SET importance=? WHERE id=?",
                (max(new_weight, 0.0), r["id"]),
            )

        conn.commit()
        conn.close()
        return below_threshold

    def purge_low_weight_fragments(self, min_weight: float = 0.1) -> int:
        """
        删除 importance 低于阈值的记忆碎片。

        ChromaDB 端采用惰性清理：已删除的碎片 ID 在后续检索时
        会因为 SQLite 中不存在而被 get_fragments_by_ids 过滤掉。

        Args:
            min_weight: 最低权重阈值，默认 0.1

        Returns:
            删除的碎片数量
        """
        conn = self._connect()
        # 先获取将被删除的 IDs（用于 ChromaDB 清理）
        to_delete = conn.execute(
            "SELECT id FROM memory_fragments WHERE importance < ?", (min_weight,)
        ).fetchall()
        deleted_ids = [str(r["id"]) for r in to_delete]

        cursor = conn.execute(
            "DELETE FROM memory_fragments WHERE importance < ?", (min_weight,)
        )
        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        # 同步清理 ChromaDB
        if deleted > 0 and deleted_ids:
            try:
                from alice.memory.chroma_client import get_memory_collection
                col = get_memory_collection()
                if col:
                    col.delete(ids=deleted_ids)
            except Exception:
                pass  # ChromaDB 不可用或 ID 不存在时忽略

        return deleted

    def get_fragment_stats(self) -> dict:
        """获取记忆碎片统计信息"""
        conn = self._connect()
        total = conn.execute("SELECT COUNT(*) FROM memory_fragments").fetchone()[0]
        avg_imp = conn.execute(
            "SELECT COALESCE(AVG(importance), 0) FROM memory_fragments"
        ).fetchone()[0]
        below = conn.execute(
            "SELECT COUNT(*) FROM memory_fragments WHERE importance < 0.1"
        ).fetchone()[0]
        by_type = {}
        for row in conn.execute(
            "SELECT fragment_type, COUNT(*) as cnt FROM memory_fragments "
            "GROUP BY fragment_type"
        ).fetchall():
            by_type[row["fragment_type"]] = row["cnt"]
        conn.close()
        return {
            "total": total,
            "avg_importance": round(avg_imp, 4),
            "below_threshold": below,
            "by_type": by_type,
        }
