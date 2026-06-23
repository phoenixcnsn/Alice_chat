"""
记忆管理器 — 编排 SQLite + ChromaDB + 摘要三层记忆系统

职责:
  1. 全量留存 — 每条消息写入 SQLite（不可物理删除）
  2. 滚动摘要 — 每 50 条消息触发一次 LLM 摘要
  3. RAG 检索 — 三路召回 + RRF 融合，返回相关记忆

注入 system prompt 的格式:
  固定人格 (角色预设) + 动态状态 (积温引擎) + 动态记忆 (RAG)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from alice.memory.store import MemoryStore, MemoryFragment, Message
from alice.memory.rag import MemoryRAG


class MemoryManager:
    """三层记忆编排器"""

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        rag: Optional[MemoryRAG] = None,
        db_path: str = "data/memory.db",
        summary_interval: int = 50,      # 每 50 条消息生成摘要
        crypto_key: Optional[bytes] = None,
    ):
        self.store = store or MemoryStore(db_path, crypto_key)
        self.rag = rag or MemoryRAG(self.store)
        self.summary_interval = summary_interval
        self.preset_name = "默认"

    # ----------------------------------------------------------------
    # 消息生命周期
    # ----------------------------------------------------------------
    def save_user_message(self, content: str, emotion_state: Dict = None) -> int:
        """保存用户消息"""
        return self.store.save_message(
            role="user", content=content, preset_name=self.preset_name,
            metadata={"emotion": emotion_state} if emotion_state else {},
        )

    def save_assistant_message(self, content: str, emotion_state: Dict = None) -> int:
        """保存助手消息"""
        return self.store.save_message(
            role="assistant", content=content, preset_name=self.preset_name,
            metadata={"emotion": emotion_state} if emotion_state else {},
        )

    # ----------------------------------------------------------------
    # 滚动摘要
    # ----------------------------------------------------------------
    async def maybe_summarize(self, llm_call) -> Optional[str]:
        """
        检查是否需要生成摘要（每 summary_interval 条消息）。
        如果需要，调用 LLM 生成并保存。
        """
        total = self.store.get_message_count(self.preset_name)
        last_summary_end = self.store.get_last_summary_end_id()

        new_since_last = total - last_summary_end
        if new_since_last < self.summary_interval:
            return None

        # 取这 50 条消息
        msgs = self.store.get_messages(limit=self.summary_interval, preset_name=self.preset_name)
        if len(msgs) < self.summary_interval:
            return None

        text = "\n".join(f"[{m.role}]: {m.content}" for m in msgs)
        start_id = msgs[0].id
        end_id = msgs[-1].id

        prompt = "请用 2-3 句话总结以下对话的关键内容和情感走向:\n\n"
        try:
            summary = await llm_call(prompt, [{"role": "user", "content": text[-4000:]}])
            self.store.save_summary(start_id, end_id, summary)
            return summary
        except Exception:
            return None

    # ----------------------------------------------------------------
    # 记忆检索（注入 system prompt 用）
    # ----------------------------------------------------------------
    def retrieve_context(self, query: str, top_n: int = 8) -> str:
        """
        检索相关记忆，返回格式化的上下文文本。

        包含:
          - 最近的滚动摘要
          - RAG 检索到的相关记忆碎片
        """
        parts = []

        # 滚动摘要
        summaries = self.store.get_all_summaries()
        if summaries:
            parts.append("## 对话历史摘要\n" + "\n".join(f"- {s}" for s in summaries[-3:]))

        # RAG 记忆碎片
        mem_text = self.rag.format_for_prompt(query, top_n)
        if mem_text:
            parts.append(mem_text)

        return "\n\n".join(parts) if parts else ""

    # ----------------------------------------------------------------
    # 异步后台提取
    # ----------------------------------------------------------------
    async def background_extract(self, llm_call):
        """后台异步提取记忆碎片（事实/偏好/情绪）"""
        await self.rag.extract_from_messages(llm_call, self.preset_name, batch_size=20)

    # ----------------------------------------------------------------
    # 记忆维护（衰减 + 清理 + 强化）
    # ----------------------------------------------------------------
    def run_maintenance(
        self,
        half_life_days: float = 30.0,
        min_weight: float = 0.1,
    ) -> dict:
        """
        执行记忆维护：衰减 → 清理低权重碎片。

        建议通过定时器周期性调用（如每天一次）。

        Args:
            half_life_days: 半衰期（天）
            min_weight: 最低权重阈值

        Returns:
            维护统计信息
        """
        # 1. 衰减
        below = self.store.decay_fragments(half_life_days, min_weight)

        # 2. 清理
        purged = self.store.purge_low_weight_fragments(min_weight)

        # 3. 统计
        stats = self.store.get_fragment_stats()

        return {
            "decayed_below_threshold": below,
            "purged": purged,
            "stats": stats,
        }

    def reinforce_memory(self, fragment_id: int, boost: float = 0.15):
        """
        强化某条记忆碎片（被检索命中或用户引用时调用）。

        Args:
            fragment_id: 碎片 ID
            boost: 强化增量 (0~1)
        """
        self.store.reinforce_fragment(fragment_id, boost)

    # ----------------------------------------------------------------
    # 获取最近消息（用于对话上下文）
    # ----------------------------------------------------------------
    def get_recent_messages(self, limit: int = 20) -> List[Message]:
        return self.store.get_messages(limit=limit, preset_name=self.preset_name)
