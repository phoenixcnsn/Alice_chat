"""
ChromaDB 向量记忆 + 三路召回 + RRF 融合排序

三路召回:
  1. FTS5 关键词    — SQLite 全文检索
  2. 向量语义        — ChromaDB embedding 相似度
  3. 实体聚合        — 按实体名匹配相关记忆

RRF (Reciprocal Rank Fusion) 融合排序后取 top-K，
动态注入 system prompt。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from alice.memory.store import MemoryStore, MemoryFragment


# ------------------------------------------------------------
# ChromaDB — 共享客户端 (alice.memory.chroma_client)
# ------------------------------------------------------------
from alice.memory.chroma_client import get_memory_collection

def _get_chroma():
    """获取记忆 ChromaDB collection（委托给共享客户端）"""
    return get_memory_collection()


# ------------------------------------------------------------
# 实体提取（基于规则 + TF-IDF 启发式）
# ------------------------------------------------------------
_COMMON_ENTITIES_PATTERN = re.compile(
    r'(?:我叫|我是|名字是|称呼我|叫我)\s*[\'"【《]?(\w{2,8})[\'"】》]?'
    r'|(?:喜欢|讨厌|爱|恨|想|要|觉得|认为|希望|打算|计划|决定)[^\n。，]{2,20}'
    r'|#[^\s#]{1,20}'
)

def extract_entities(text: str) -> List[str]:
    """从文本中提取实体（规则 + 启发式）"""
    entities = []
    for m in _COMMON_ENTITIES_PATTERN.finditer(text):
        entity = m.group(0).strip()[:30]
        if entity and len(entity) >= 1:
            entities.append(entity)
    # 去重，限制数量
    seen = set()
    result = []
    for e in entities:
        if e not in seen:
            seen.add(e)
            result.append(e)
    return result[:10]


# ------------------------------------------------------------
# RRF 融合排序
# ------------------------------------------------------------
def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[int, float]]],
    k: int = 60,
    top_n: int = 10,
) -> List[Tuple[int, float]]:
    """
    RRF 融合多路排序结果。

    Args:
        ranked_lists: 每路结果 [(fragment_id, score), ...]
        k: RRF 平滑参数
        top_n: 返回数量

    Returns:
        [(fragment_id, fused_score), ...] 按融合分降序
    """
    scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (fid, _) in enumerate(ranked, start=1):
            scores[fid] = scores.get(fid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


# ------------------------------------------------------------
# MemoryRAG
# ------------------------------------------------------------
class MemoryRAG:
    """向量记忆检索 + 异步提取"""

    def __init__(self, store: MemoryStore, embed_fn=None):
        """
        Args:
            store: MemoryStore 实例
            embed_fn: embedding 函数 async (texts: List[str]) -> List[List[float]]
                      不提供则回退到纯 FTS5 + 实体检索
        """
        self.store = store
        self.embed_fn = embed_fn
        self._last_extraction_msg_id = 0
        self._extraction_batch: List[int] = []

    # ----------------------------------------------------------------
    # 向量化 & 索引
    # ----------------------------------------------------------------
    def _vectorize_and_index(self, fragment: MemoryFragment):
        """将记忆碎片向量化并存入 ChromaDB"""
        if not self.embed_fn:
            return
        col = _get_chroma()
        if not col:
            return
        try:
            from chromadb.api.types import Embedding
            emb = self.embed_fn([fragment.content])
            if emb and emb[0]:
                col.upsert(
                    ids=[str(fragment.id)],
                    embeddings=[emb[0]],
                    documents=[fragment.content],
                    metadatas=[{
                        "fragment_type": fragment.fragment_type,
                        "entities": ",".join(fragment.entities),
                        "importance": fragment.importance,
                    }],
                )
        except Exception:
            pass  # embedding 失败不影响主流程

    # ----------------------------------------------------------------
    # 三路召回
    # ----------------------------------------------------------------
    def retrieve(self, query: str, top_n: int = 10) -> List[MemoryFragment]:
        """
        三路召回 + RRF 融合。

        Args:
            query: 用户当前消息
            top_n: 返回条数

        Returns:
            排序后的记忆碎片列表
        """
        ranked_lists: List[List[Tuple[int, float]]] = []

        # ---- 路1: FTS5 关键词检索 ----
        fts_ids = self.store.search_fts(query, limit=top_n * 2)
        ranked_lists.append([(fid, float(i)) for i, fid in enumerate(fts_ids)])

        # ---- 路2: 实体聚合检索 ----
        entities = extract_entities(query)
        entity_frags = self.store.get_fragments_by_entities(entities, limit=top_n)
        ranked_lists.append([(f.id, f.importance) for f in entity_frags])

        # ---- 路3: 向量语义检索 ----
        if self.embed_fn:
            col = _get_chroma()
            if col and col.count() > 0:
                try:
                    q_emb = self.embed_fn([query])
                    if q_emb and q_emb[0]:
                        results = col.query(
                            query_embeddings=[q_emb[0]],
                            n_results=top_n,
                        )
                        vector_ids = results.get("ids", [[]])[0]
                        vector_dists = results.get("distances", [[]])[0]
                        ranked_lists.append([
                            (int(vid), 1.0 - float(vd))
                            for vid, vd in zip(vector_ids, vector_dists)
                        ])
                except Exception:
                    pass

        # ---- RRF 融合 ----
        fused = reciprocal_rank_fusion(ranked_lists, top_n=top_n)
        frags = self.store.get_fragments_by_ids([fid for fid, _ in fused])

        # 保持 RRF 顺序
        score_map = dict(fused)
        frags.sort(key=lambda f: score_map.get(f.id, 0), reverse=True)

        # 强化被检索到的记忆（微小 boost，降低衰减速度）
        for f in frags[:top_n]:
            try:
                self.store.reinforce_fragment(f.id, boost=0.02)  # 检索 boost 很小
            except Exception:
                pass

        return frags[:top_n]

    # ----------------------------------------------------------------
    # 异步提取（后台 LLM 分析）
    # ----------------------------------------------------------------
    async def extract_from_messages(
        self, llm_call, preset_name: str = "", batch_size: int = 20
    ):
        """
        从新消息中异步提取事实/偏好/情绪碎片。

        每 batch_size 条消息触发一次提取。
        """
        new_msgs = self.store.get_messages_since(self._last_extraction_msg_id, preset_name)
        if not new_msgs:
            return

        self._extraction_batch.extend([m.id for m in new_msgs])
        self._last_extraction_msg_id = max(m.id for m in new_msgs)

        if len(self._extraction_batch) < batch_size:
            return

        # 取一批消息
        batch_ids = self._extraction_batch[:batch_size]
        self._extraction_batch = self._extraction_batch[batch_size:]

        # 从 messages 表查询原始消息文本
        conn = self.store._connect()
        placeholders = ",".join("?" * len(batch_ids))
        rows = conn.execute(
            f"SELECT * FROM messages WHERE id IN ({placeholders}) ORDER BY id",
            batch_ids
        ).fetchall()
        conn.close()

        msg_texts = []
        for r in rows:
            content = self.store._decrypt(r["content"])
            msg_texts.append(f"[{r['role']}]: {content}")

        if not msg_texts or not llm_call:
            return

        # LLM 提取 prompt
        extract_prompt = """分析以下对话，提取关键记忆碎片。返回 JSON 数组。

每条碎片格式: {"type": "fact|preference|emotion|event", "content": "碎片内容", "entities": ["实体1"], "importance": 0.0-1.0}

规则:
- fact: 客观事实 (用户说过的个人信息、事件)
- preference: 偏好 (喜欢/讨厌什么)
- emotion: 情绪片段 (用户表达的情绪状态)
- event: 对话中发生的事件
- importance: 重要性 0-1, 1=非常关键的信息
- entities: 相关实体名列表

对话:
"""
        full_text = "\n".join(msg_texts[-30:])  # 最多取最近 30 条

        try:
            response = await llm_call(extract_prompt, [
                {"role": "user", "content": full_text}
            ])

            # 解析 JSON
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                items = json.loads(json_match.group(0))
                for item in items:
                    fragment = MemoryFragment(
                        fragment_type=item.get("type", "fact"),
                        content=item.get("content", ""),
                        source_msg_ids=batch_ids,
                        entities=item.get("entities", []),
                        importance=float(item.get("importance", 0.5)),
                    )
                    fid = self.store.save_fragment(fragment)
                    fragment.id = fid
                    self._vectorize_and_index(fragment)
        except Exception:
            pass  # 提取失败不影响主对话

    # ----------------------------------------------------------------
    # 记忆格式化（注入 system prompt）
    # ----------------------------------------------------------------
    def format_memories(self, fragments: List[MemoryFragment]) -> str:
        """将记忆碎片格式化为可注入 system prompt 的文本"""
        if not fragments:
            return ""

        lines = ["## 相关记忆"]
        for f in fragments[:8]:
            tag = {"fact": "📋", "preference": "💝", "emotion": "💭", "event": "📅"}.get(
                f.fragment_type, "📌")
            entities_str = f" [{', '.join(f.entities)}]" if f.entities else ""
            lines.append(f"{tag} {f.content}{entities_str}")
        return "\n".join(lines)

    def format_for_prompt(self, query: str, top_n: int = 8) -> str:
        """检索 + 格式化，一步完成"""
        frags = self.retrieve(query, top_n)
        return self.format_memories(frags)
