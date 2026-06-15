"""
风格向量存储 — 基于 ChromaDB 的角色对话示例索引

存储从源文本中提取的对话示例（user→character 对），
运行时根据当前对话上下文检索最相关的风格示例，
作为 few-shot 注入 system prompt。

与 memory_store.py 的区别:
  - memory_store: 存储运行时对话记忆（事实/偏好/情绪）
  - style_store:  存储角色源文本中的风格示例（对话片段/说话方式）

存储维度:
  1. ChromaDB:    对话示例的语义向量（用于相似度检索）
  2. SQLite:      原始对话示例 + 元数据（来源文件、批次、标签）
  3. 统计指纹:    句长分布、标点模式、词汇频率等（非 LLM 提取）
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------------------------------------
# ChromaDB (lazy import — 与 memory_rag.py 共享同一模式)
# ------------------------------------------------------------
_STYLE_CHROMA_AVAILABLE = False
_style_chroma_client = None
_style_collection = None


def _get_style_chroma(path: str = "data/style_chroma"):
    """获取风格向量存储的 ChromaDB collection"""
    global _STYLE_CHROMA_AVAILABLE, _style_chroma_client, _style_collection
    if not _STYLE_CHROMA_AVAILABLE:
        try:
            import chromadb
            Path(path).mkdir(parents=True, exist_ok=True)
            _style_chroma_client = chromadb.PersistentClient(path=path)
            _style_collection = _style_chroma_client.get_or_create_collection(
                name="character_style_examples",
                metadata={"hnsw:space": "cosine"},
            )
            _STYLE_CHROMA_AVAILABLE = True
        except ImportError:
            pass
    return _style_collection


# ------------------------------------------------------------
# 数据模型
# ------------------------------------------------------------
@dataclass
class StyleExample:
    """一条风格示例 — 从源文本中提取的对话片段"""
    id: int = 0
    preset_name: str = ""                   # 所属角色预设
    source_file: str = ""                   # 来源文件名
    batch_id: str = ""                      # 批次 ID（增量添加时区分）
    context: str = ""                       # 对话上下文（用户说了什么 / 场景描述）
    character_response: str = ""            # 角色的典型回复
    tags: List[str] = field(default_factory=list)   # 标签: "傲娇", "关心", "生气" 等
    quality_score: float = 0.5              # 质量分 (0-1)，用于排序
    created_at: float = field(default_factory=time.time)


@dataclass
class StyleFingerprint:
    """统计风格指纹 — 非 LLM 提取的表面语言特征"""
    preset_name: str = ""
    # 句长分布
    avg_sentence_length: float = 0.0
    sentence_length_std: float = 0.0
    short_sentence_ratio: float = 0.0       # 短句占比 (<10字)
    long_sentence_ratio: float = 0.0        # 长句占比 (>30字)
    # 标点偏好
    exclamation_ratio: float = 0.0          # 感叹号频率
    question_ratio: float = 0.0             # 问号频率
    ellipsis_ratio: float = 0.0             # 省略号频率
    comma_ratio: float = 0.0               # 逗号频率
    # 词汇特征
    top_words: List[str] = field(default_factory=list)        # 高频词 top-20
    top_bigrams: List[str] = field(default_factory=list)      # 高频二元组 top-10
    common_starters: List[str] = field(default_factory=list)  # 常见句首
    common_enders: List[str] = field(default_factory=list)    # 常见句尾
    # 语气特征
    particle_ratio: float = 0.0             # 语气词占比 (啊/呀/呢/吧/嘛/哦/哈/哎)
    interjection_ratio: float = 0.0         # 感叹词占比 (哇/哎呀/哼/切/啧)
    # 统计来源
    total_chars_analyzed: int = 0
    total_sentences_analyzed: int = 0


# ------------------------------------------------------------
# SQLite 存储层
# ------------------------------------------------------------
STYLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS style_examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    preset_name TEXT NOT NULL,
    source_file TEXT DEFAULT '',
    batch_id TEXT DEFAULT '',
    context TEXT NOT NULL,
    character_response TEXT NOT NULL,
    tags TEXT DEFAULT '[]',
    quality_score REAL DEFAULT 0.5,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS style_fingerprints (
    preset_name TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS corpus_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    preset_name TEXT NOT NULL,
    source_file TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    char_count INTEGER DEFAULT 0,
    example_count INTEGER DEFAULT 0,
    added_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_style_preset ON style_examples(preset_name);
CREATE INDEX IF NOT EXISTS idx_style_batch ON style_examples(batch_id);
CREATE INDEX IF NOT EXISTS idx_style_quality ON style_examples(quality_score);
CREATE INDEX IF NOT EXISTS idx_corpus_preset ON corpus_sources(preset_name);
"""


from alice.storage.base import SQLiteStore


class StyleStore(SQLiteStore):
    """角色风格示例的 SQLite + ChromaDB 双层存储"""

    def __init__(self, db_path: str = "data/style_store.db",
                 chroma_path: str = "data/style_chroma",
                 embed_fn=None):
        """
        Args:
            db_path: SQLite 数据库路径
            chroma_path: ChromaDB 持久化路径
            embed_fn: embedding 函数 async (texts: List[str]) -> List[List[float]]
        """
        super().__init__(db_path)
        self.chroma_path = chroma_path
        self.embed_fn = embed_fn
        self._init_db()

    def _init_db(self):
        conn = self._connect()
        conn.executescript(STYLE_SCHEMA)
        conn.commit()
        conn.close()

    # ----------------------------------------------------------------
    # 风格示例 CRUD
    # ----------------------------------------------------------------
    def add_example(self, example: StyleExample) -> int:
        """添加一条风格示例，返回 ID"""
        conn = self._connect()
        conn.execute(
            """INSERT INTO style_examples
               (preset_name, source_file, batch_id, context, character_response,
                tags, quality_score, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (example.preset_name, example.source_file, example.batch_id,
             example.context, example.character_response,
             json.dumps(example.tags, ensure_ascii=False),
             example.quality_score, example.created_at or time.time())
        )
        conn.commit()
        eid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()

        # 同步到 ChromaDB
        self._index_example(example, eid)

        return eid

    def add_examples_batch(self, examples: List[StyleExample]) -> List[int]:
        """批量添加风格示例，返回 ID 列表"""
        ids = []
        conn = self._connect()
        now = time.time()
        for ex in examples:
            conn.execute(
                """INSERT INTO style_examples
                   (preset_name, source_file, batch_id, context, character_response,
                    tags, quality_score, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (ex.preset_name, ex.source_file, ex.batch_id,
                 ex.context, ex.character_response,
                 json.dumps(ex.tags, ensure_ascii=False),
                 ex.quality_score, ex.created_at or now)
            )
        conn.commit()
        # 获取所有新 ID
        last_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()

        # 计算起始 ID
        start_id = last_id - len(examples) + 1
        all_ids = list(range(start_id, last_id + 1))

        # 批量索引到 ChromaDB
        for ex, eid in zip(examples, all_ids):
            ex.id = eid
            self._index_example(ex, eid)

        return all_ids

    def _index_example(self, example: StyleExample, eid: int):
        """将单条示例向量化并存入 ChromaDB"""
        if not self.embed_fn:
            return
        col = _get_style_chroma(self.chroma_path)
        if not col:
            return
        try:
            # 将 context + response 拼接为检索文本
            search_text = f"用户: {example.context}\n角色: {example.character_response}"
            emb = self.embed_fn([search_text])
            if emb and emb[0]:
                col.upsert(
                    ids=[str(eid)],
                    embeddings=[emb[0]],
                    documents=[search_text],
                    metadatas=[{
                        "preset_name": example.preset_name,
                        "tags": ",".join(example.tags),
                        "quality_score": example.quality_score,
                    }],
                )
        except Exception:
            pass  # embedding 失败不影响主流程

    def get_examples_by_preset(self, preset_name: str, limit: int = 100) -> List[StyleExample]:
        """获取指定预设的所有风格示例"""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM style_examples WHERE preset_name=? ORDER BY quality_score DESC LIMIT ?",
            (preset_name, limit)
        ).fetchall()
        conn.close()
        return [StyleExample(
            id=r["id"], preset_name=r["preset_name"],
            source_file=r["source_file"], batch_id=r["batch_id"],
            context=r["context"], character_response=r["character_response"],
            tags=json.loads(r["tags"]), quality_score=r["quality_score"],
            created_at=r["created_at"]
        ) for r in rows]

    def get_example_count(self, preset_name: str) -> int:
        conn = self._connect()
        row = conn.execute(
            "SELECT COUNT(*) FROM style_examples WHERE preset_name=?", (preset_name,)
        ).fetchone()
        conn.close()
        return row[0]

    def delete_preset_examples(self, preset_name: str):
        """删除预设的所有风格示例（用于重新训练）"""
        conn = self._connect()
        conn.execute("DELETE FROM style_examples WHERE preset_name=?", (preset_name,))
        conn.commit()
        conn.close()
        # 同时清理 ChromaDB（通过重新 collection 实现，或标记删除）
        col = _get_style_chroma(self.chroma_path)
        if col:
            try:
                results = col.get(where={"preset_name": preset_name})
                if results and results["ids"]:
                    col.delete(ids=results["ids"])
            except Exception:
                pass

    # ----------------------------------------------------------------
    # 风格指纹
    # ----------------------------------------------------------------
    def save_fingerprint(self, fp: StyleFingerprint):
        conn = self._connect()
        data = json.dumps(self._fingerprint_to_dict(fp), ensure_ascii=False)
        conn.execute(
            """INSERT OR REPLACE INTO style_fingerprints (preset_name, data, updated_at)
               VALUES (?,?,?)""",
            (fp.preset_name, data, time.time())
        )
        conn.commit()
        conn.close()

    def load_fingerprint(self, preset_name: str) -> Optional[StyleFingerprint]:
        conn = self._connect()
        row = conn.execute(
            "SELECT data FROM style_fingerprints WHERE preset_name=?", (preset_name,)
        ).fetchone()
        conn.close()
        if row:
            return self._dict_to_fingerprint(json.loads(row["data"]))
        return None

    @staticmethod
    def _fingerprint_to_dict(fp: StyleFingerprint) -> Dict:
        return {
            "preset_name": fp.preset_name,
            "avg_sentence_length": fp.avg_sentence_length,
            "sentence_length_std": fp.sentence_length_std,
            "short_sentence_ratio": fp.short_sentence_ratio,
            "long_sentence_ratio": fp.long_sentence_ratio,
            "exclamation_ratio": fp.exclamation_ratio,
            "question_ratio": fp.question_ratio,
            "ellipsis_ratio": fp.ellipsis_ratio,
            "comma_ratio": fp.comma_ratio,
            "top_words": fp.top_words,
            "top_bigrams": fp.top_bigrams,
            "common_starters": fp.common_starters,
            "common_enders": fp.common_enders,
            "particle_ratio": fp.particle_ratio,
            "interjection_ratio": fp.interjection_ratio,
            "total_chars_analyzed": fp.total_chars_analyzed,
            "total_sentences_analyzed": fp.total_sentences_analyzed,
        }

    @staticmethod
    def _dict_to_fingerprint(d: Dict) -> StyleFingerprint:
        return StyleFingerprint(
            preset_name=d.get("preset_name", ""),
            avg_sentence_length=d.get("avg_sentence_length", 0.0),
            sentence_length_std=d.get("sentence_length_std", 0.0),
            short_sentence_ratio=d.get("short_sentence_ratio", 0.0),
            long_sentence_ratio=d.get("long_sentence_ratio", 0.0),
            exclamation_ratio=d.get("exclamation_ratio", 0.0),
            question_ratio=d.get("question_ratio", 0.0),
            ellipsis_ratio=d.get("ellipsis_ratio", 0.0),
            comma_ratio=d.get("comma_ratio", 0.0),
            top_words=d.get("top_words", []),
            top_bigrams=d.get("top_bigrams", []),
            common_starters=d.get("common_starters", []),
            common_enders=d.get("common_enders", []),
            particle_ratio=d.get("particle_ratio", 0.0),
            interjection_ratio=d.get("interjection_ratio", 0.0),
            total_chars_analyzed=d.get("total_chars_analyzed", 0),
            total_sentences_analyzed=d.get("total_sentences_analyzed", 0),
        )

    # ----------------------------------------------------------------
    # 语料来源追踪
    # ----------------------------------------------------------------
    def record_corpus(self, preset_name: str, source_file: str, batch_id: str,
                      char_count: int, example_count: int):
        conn = self._connect()
        conn.execute(
            """INSERT INTO corpus_sources (preset_name, source_file, batch_id,
               char_count, example_count, added_at)
               VALUES (?,?,?,?,?,?)""",
            (preset_name, source_file, batch_id, char_count, example_count, time.time())
        )
        conn.commit()
        conn.close()

    def get_corpus_sources(self, preset_name: str) -> List[Dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM corpus_sources WHERE preset_name=? ORDER BY added_at",
            (preset_name,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_total_corpus_chars(self, preset_name: str) -> int:
        conn = self._connect()
        row = conn.execute(
            "SELECT COALESCE(SUM(char_count), 0) FROM corpus_sources WHERE preset_name=?",
            (preset_name,)
        ).fetchone()
        conn.close()
        return row[0]

    # ----------------------------------------------------------------
    # 语义检索（核心：运行时检索相似风格示例）
    # ----------------------------------------------------------------
    def retrieve_style_examples(
        self, query: str, preset_name: str = "", top_n: int = 5
    ) -> List[StyleExample]:
        """
        根据当前对话上下文，检索最相关的风格示例。

        两路检索:
          1. ChromaDB 语义相似度（主）
          2. 标签关键词匹配（辅，无 embedding 时回退）

        Args:
            query: 当前对话上下文（用户消息 + 最近的对话）
            preset_name: 限定角色预设（为空则不限）
            top_n: 返回条数

        Returns:
            按相关度排序的风格示例列表
        """
        col = _get_style_chroma(self.chroma_path)

        # ---- 路径 1: ChromaDB 语义检索 ----
        if col and col.count() > 0 and self.embed_fn:
            try:
                q_emb = self.embed_fn([query])
                if q_emb and q_emb[0]:
                    where_filter = {"preset_name": preset_name} if preset_name else None
                    results = col.query(
                        query_embeddings=[q_emb[0]],
                        n_results=top_n * 2,
                        where=where_filter,
                    )
                    result_ids = results.get("ids", [[]])[0]
                    if result_ids:
                        ids = [int(rid) for rid in result_ids]
                        examples = self._get_examples_by_ids(ids)
                        # 按 ChromaDB 返回顺序 + quality_score 排序
                        id_order = {eid: i for i, eid in enumerate(ids)}
                        examples.sort(key=lambda e: (
                            id_order.get(e.id, 999),
                            -e.quality_score
                        ))
                        return examples[:top_n]
            except Exception:
                pass

        # ---- 路径 2: 关键词回退 ----
        return self._keyword_retrieve(query, preset_name, top_n)

    def _keyword_retrieve(self, query: str, preset_name: str, top_n: int) -> List[StyleExample]:
        """关键词回退检索"""
        conn = self._connect()
        # 提取关键词（中文单字+双字组合）
        chars = re.sub(r'[^一-鿿]', '', query)
        keywords = []
        for i in range(len(chars)):
            keywords.append(chars[i])
            if i + 1 < len(chars):
                keywords.append(chars[i:i+2])

        if not keywords:
            keywords = [query[:5]]

        found_ids = []
        seen = set()
        for kw in keywords[:15]:
            if preset_name:
                rows = conn.execute(
                    """SELECT id, quality_score FROM style_examples
                       WHERE preset_name=? AND (context LIKE ? OR character_response LIKE ?)
                       ORDER BY quality_score DESC LIMIT ?""",
                    (preset_name, f"%{kw}%", f"%{kw}%", max(1, top_n // len(keywords)))
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT id, quality_score FROM style_examples
                       WHERE context LIKE ? OR character_response LIKE ?
                       ORDER BY quality_score DESC LIMIT ?""",
                    (f"%{kw}%", f"%{kw}%", max(1, top_n // len(keywords)))
                ).fetchall()
            for r in rows:
                if r["id"] not in seen:
                    seen.add(r["id"])
                    found_ids.append(r["id"])
        conn.close()

        return self._get_examples_by_ids(found_ids[:top_n])

    def _get_examples_by_ids(self, ids: List[int]) -> List[StyleExample]:
        if not ids:
            return []
        conn = self._connect()
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT * FROM style_examples WHERE id IN ({placeholders})",
            ids
        ).fetchall()
        conn.close()

        examples = [StyleExample(
            id=r["id"], preset_name=r["preset_name"],
            source_file=r["source_file"], batch_id=r["batch_id"],
            context=r["context"], character_response=r["character_response"],
            tags=json.loads(r["tags"]), quality_score=r["quality_score"],
            created_at=r["created_at"]
        ) for r in rows]

        # 保持传入 ID 顺序
        id_order = {eid: i for i, eid in enumerate(ids)}
        examples.sort(key=lambda e: id_order.get(e.id, 999))
        return examples


# ------------------------------------------------------------
# 统计风格指纹提取（非 LLM，纯规则+统计）
# ------------------------------------------------------------
def extract_style_fingerprint(texts: List[str], preset_name: str = "") -> StyleFingerprint:
    """
    从角色文本中提取统计风格指纹。

    纯规则算法，不需要 LLM，速度快，可随时增量更新。

    Args:
        texts: 角色的对话文本列表（每条是一段角色说的话）
        preset_name: 预设名

    Returns:
        StyleFingerprint
    """
    fp = StyleFingerprint(preset_name=preset_name)

    if not texts:
        return fp

    # 合并所有文本
    all_text = "\n".join(texts)
    fp.total_chars_analyzed = len(all_text)

    # 分句（按中文标点切分）
    sentences = re.split(r'[。！？!?\n]+', all_text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) >= 2]
    fp.total_sentences_analyzed = len(sentences)

    if not sentences:
        return fp

    # 句长分析
    sent_lengths = [len(s) for s in sentences]
    fp.avg_sentence_length = sum(sent_lengths) / len(sent_lengths)
    fp.sentence_length_std = (
        sum((l - fp.avg_sentence_length) ** 2 for l in sent_lengths) / len(sent_lengths)
    ) ** 0.5
    fp.short_sentence_ratio = sum(1 for l in sent_lengths if l < 10) / len(sent_lengths)
    fp.long_sentence_ratio = sum(1 for l in sent_lengths if l > 30) / len(sent_lengths)

    # 标点频率（以字符计）
    total_chars = len(all_text)
    if total_chars > 0:
        fp.exclamation_ratio = all_text.count('！') + all_text.count('!') / total_chars
        fp.question_ratio = all_text.count('？') + all_text.count('?') / total_chars
        fp.ellipsis_ratio = (all_text.count('…') + all_text.count('...')) / total_chars
        fp.comma_ratio = all_text.count('，') + all_text.count(',') / total_chars

    # 词汇频率（中文 1-2 gram）
    # 去除标点和空白后的纯中文
    pure_chinese = re.sub(r'[^一-鿿]', '', all_text)
    if pure_chinese:
        unigrams = [pure_chinese[i] for i in range(len(pure_chinese))]
        bigrams = [pure_chinese[i:i+2] for i in range(len(pure_chinese)-1)]

        word_counter = Counter(unigrams)
        # 过滤单字停用词
        stop_chars = set('的是我了在有不这他为个们以一就上也到得说要去会可你')
        fp.top_words = [w for w, _ in word_counter.most_common(30)
                       if w not in stop_chars][:20]

        bigram_counter = Counter(bigrams)
        fp.top_bigrams = [bg for bg, _ in bigram_counter.most_common(10)]

    # 句首/句尾词
    starters = []
    enders = []
    for s in sentences[:200]:  # 取前 200 句分析
        if len(s) >= 3:
            starters.append(s[:2])
            enders.append(s[-2:])
    fp.common_starters = [s for s, c in Counter(starters).most_common(8) if c >= 2]
    fp.common_enders = [e for e, c in Counter(enders).most_common(8) if c >= 2]

    # 语气词和感叹词频率
    particles = set('啊呀呢吧嘛哦哈哎嗯啦唷呐')
    interjections = set('哇哎呀哼切啧咦嘿哟嗷')
    if total_chars > 0:
        fp.particle_ratio = sum(1 for c in all_text if c in particles) / total_chars
        fp.interjection_ratio = sum(1 for c in all_text if c in interjections) / total_chars

    return fp


def merge_fingerprints(old: StyleFingerprint, new: StyleFingerprint,
                       old_weight: float = 0.5) -> StyleFingerprint:
    """
    合并两个风格指纹（增量学习时使用）。

    新指纹的权重 = 1 - old_weight。
    对于列表字段（top_words 等），做加权合并。
    """
    w_new = 1.0 - old_weight
    merged = StyleFingerprint(preset_name=old.preset_name or new.preset_name)

    # 数值字段加权平均
    for field_name in [
        'avg_sentence_length', 'sentence_length_std',
        'short_sentence_ratio', 'long_sentence_ratio',
        'exclamation_ratio', 'question_ratio', 'ellipsis_ratio', 'comma_ratio',
        'particle_ratio', 'interjection_ratio',
    ]:
        old_val = getattr(old, field_name, 0.0)
        new_val = getattr(new, field_name, 0.0)
        setattr(merged, field_name, old_val * old_weight + new_val * w_new)

    # 列表字段: 合并 + 重新排序
    for field_name in ['top_words', 'top_bigrams', 'common_starters', 'common_enders']:
        old_list = getattr(old, field_name, [])
        new_list = getattr(new, field_name, [])
        combined = {}
        for i, item in enumerate(old_list):
            combined[item] = combined.get(item, 0) + (len(old_list) - i) * old_weight
        for i, item in enumerate(new_list):
            combined[item] = combined.get(item, 0) + (len(new_list) - i) * w_new
        setattr(merged, field_name, [
            item for item, _ in sorted(combined.items(), key=lambda x: x[1], reverse=True)[:20]
        ])

    merged.total_chars_analyzed = old.total_chars_analyzed + new.total_chars_analyzed
    merged.total_sentences_analyzed = old.total_sentences_analyzed + new.total_sentences_analyzed

    return merged
