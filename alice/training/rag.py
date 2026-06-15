"""
人格 RAG — 运行时风格检索 + 动态提示词组装

核心思路:
  不再用一段静态文本描述角色的说话风格，
  而是在每次对话时，从源文本中检索最相关的真实对话示例，
  作为 few-shot 注入 system prompt。

与 memory_rag.py 的分工:
  - memory_rag: 检索"过去的对话记忆"（事实/偏好/情绪）
  - personality_rag: 检索"角色的风格示例"（源文本中的对话片段）

注入 system prompt 的层级结构:
  Layer 1: 核心身份  (来自 CharacterProfile, 固定)
  Layer 2: 风格示例  (来自 StyleStore RAG, 动态)  ← 新增
  Layer 3: 情绪状态  (来自积温引擎, 动态)
  Layer 4: 对话记忆  (来自 MemoryRAG, 动态)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from alice.training.styles import StyleStore, StyleExample, StyleFingerprint


class PersonalityRAG:
    """人格风格 RAG — 运行时检索 + 多层级 prompt 组装"""

    def __init__(self, style_store: StyleStore):
        self.store = style_store

    # ----------------------------------------------------------------
    # 检索
    # ----------------------------------------------------------------
    def retrieve_style_examples(
        self,
        current_context: str,
        preset_name: str,
        top_n: int = 5,
        min_quality: float = 0.0,
    ) -> List[StyleExample]:
        """
        检索与当前对话上下文最相关的风格示例。

        Args:
            current_context: 当前对话上下文（用户刚说的 + 最近几轮对话摘要）
            preset_name: 角色预设名
            top_n: 返回数量
            min_quality: 最低质量分阈值

        Returns:
            排序后的风格示例列表
        """
        examples = self.store.retrieve_style_examples(
            query=current_context,
            preset_name=preset_name,
            top_n=top_n * 2,  # 多取一些，过滤后保留 top_n
        )

        # 按质量分过滤
        if min_quality > 0:
            examples = [e for e in examples if e.quality_score >= min_quality]

        # 去重相似示例（简单去重：character_response 前 10 字相同视为重复）
        seen = set()
        unique = []
        for e in examples:
            key = e.character_response[:10]
            if key not in seen:
                seen.add(key)
                unique.append(e)
                if len(unique) >= top_n:
                    break

        return unique

    # ----------------------------------------------------------------
    # 风格示例格式化（注入 system prompt）
    # ----------------------------------------------------------------
    def format_style_examples(self, examples: List[StyleExample]) -> str:
        """
        将风格示例格式化为 few-shot 样式，注入 system prompt。

        返回的文本可以直接追加到 system prompt 末尾。
        """
        if not examples:
            return ""

        lines = [
            "",
            "## 角色对话风格参考（来自源文本的真实对话示例）",
            "以下是你（角色）在类似情境下的真实回复。请参考这些示例的**语气、句式、用词习惯**来回复，",
            "但不要直接复制。保持风格一致，内容根据当前对话调整。",
            "",
        ]

        for i, ex in enumerate(examples, 1):
            tags_str = f" [{', '.join(ex.tags)}]" if ex.tags else ""
            lines.append(f"### 示例 {i}{tags_str}")
            lines.append(f"**对方说**: {ex.context}")
            lines.append(f"**你回复**: {ex.character_response}")
            lines.append("")

        return "\n".join(lines)

    def format_style_fingerprint(self, fp: StyleFingerprint) -> str:
        """
        将风格指纹格式化为简洁的写作指引，注入 system prompt。

        与 style_examples 互补 — 示例给出具体参考，指纹给出总体模式。
        """
        if not fp or fp.total_sentences_analyzed < 5:
            return ""

        lines = [
            "",
            "## 语言风格数据（统计特征）",
            f"- 平均句长: {fp.avg_sentence_length:.1f} 字，标准差: {fp.sentence_length_std:.1f}",
        ]

        if fp.short_sentence_ratio > 0.3:
            lines.append(f"- 短句偏好: 高 ({fp.short_sentence_ratio:.0%} 的句子 < 10 字)")
        if fp.long_sentence_ratio > 0.2:
            lines.append(f"- 长句偏好: 较高 ({fp.long_sentence_ratio:.0%} 的句子 > 30 字)")

        if fp.exclamation_ratio > 0.01:
            lines.append(f"- 感叹号使用: 频繁 ({fp.exclamation_ratio*100:.1f}%)")
        if fp.question_ratio > 0.01:
            lines.append(f"- 问号使用: 频繁 ({fp.question_ratio*100:.1f}%)")
        if fp.ellipsis_ratio > 0.005:
            lines.append(f"- 省略号使用: 较多 ({fp.ellipsis_ratio*100:.1f}%)")

        if fp.particle_ratio > 0.02:
            lines.append(f"- 语气词丰富: 高频使用啊/呀/呢/吧/嘛等")
        if fp.interjection_ratio > 0.005:
            lines.append(f"- 感叹词丰富: 高频使用哇/哎呀/哼等")

        if fp.common_starters:
            starters = "、".join(fp.common_starters[:5])
            lines.append(f"- 常见句首: {starters}")
        if fp.common_enders:
            enders = "、".join(fp.common_enders[:5])
            lines.append(f"- 常见句尾: {enders}")

        if fp.top_words:
            words = "、".join(fp.top_words[:10])
            lines.append(f"- 高频用字: {words}")

        return "\n".join(lines) + "\n"

    # ----------------------------------------------------------------
    # 构建完整的多层级 system prompt
    # ----------------------------------------------------------------
    def build_enriched_system_prompt(
        self,
        base_prompt: str,
        preset_name: str,
        current_context: str,
        emotion_context: str = "",
        memory_context: str = "",
        style_example_count: int = 5,
    ) -> str:
        """
        组装完整的多层级系统提示词。

        层级:
          1. 核心人格 (base_prompt)
          2. 记忆上下文 (memory_context) — 可选
          3. 情绪状态 (emotion_context) — 可选
          4. 风格示例 RAG (动态检索)

        Args:
            base_prompt: 来自 CharacterProfile.build_system_prompt() 的基础提示词
            preset_name: 角色预设名
            current_context: 当前对话上下文
            emotion_context: 积温引擎情绪上下文
            memory_context: 记忆 RAG 上下文
            style_example_count: 检索的风格示例数量

        Returns:
            完整的 system prompt
        """
        parts = [base_prompt]

        # Layer 2: 记忆上下文
        if memory_context:
            parts.append(memory_context)

        # Layer 3: 情绪状态
        if emotion_context:
            parts.append(f"\n## 当前感受\n{emotion_context}")

        # Layer 4: 风格 RAG
        examples = self.retrieve_style_examples(
            current_context=current_context,
            preset_name=preset_name,
            top_n=style_example_count,
        )

        if examples:
            style_text = self.format_style_examples(examples)
            parts.append(style_text)

        # 可选: 风格指纹
        fp = self.store.load_fingerprint(preset_name)
        if fp and fp.total_sentences_analyzed >= 5:
            fp_text = self.format_style_fingerprint(fp)
            if fp_text:
                # 如果风格示例少，指纹更重要；反之则略简洁
                if len(examples) < 3:
                    parts.append(fp_text)
                else:
                    # 只放最关键的 2-3 条
                    fp_lines = fp_text.split("\n")
                    parts.append("\n".join(fp_lines[:5]))

        return "\n".join(parts)

    # ----------------------------------------------------------------
    # 检索对话历史中角色自己的回复（用于自我一致性）
    # ----------------------------------------------------------------
    def retrieve_self_consistency_examples(
        self, query: str, preset_name: str, top_n: int = 3
    ) -> List[StyleExample]:
        """
        检索角色在历史对话中的回复，用于保持一致性。

        这类似于 RAG 版本的"角色记忆"——角色记得自己之前怎么说的。
        """
        return self.retrieve_style_examples(
            current_context=query,
            preset_name=preset_name,
            top_n=top_n,
        )
