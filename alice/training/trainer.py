"""
人格训练器 — 多阶段角色人格学习管道

将文本到人格的转换建模为类似 LLM 后训练的过程:

  Phase 1: 语料预处理   — 分句、分角色、分段
  Phase 2: 多维度提取   — 统计指纹 + 对话采样 + 风格提取 + 参数映射
  Phase 3: 向量索引     — 所有示例存入 ChromaDB
  Phase 4: 增量融合     — 新文本合并到已有特征（加权平均）
  Phase 5: 保存存档     — 完整 checkpoint 保存

类比:
  - Pre-training  = Phase 1+2 (首次从文本构建人格基础)
  - SFT           = Phase 2 中的对话采样 (提取高质量对话示例)
  - LoRA/PEFT     = Phase 4 (增量添加文本，只更新受影响的部分)
  - Checkpoint    = Phase 5 (保存完整状态)

用法:
    from alice.training.trainer import PersonalityTrainer

    trainer = PersonalityTrainer(llm_call=my_llm_func)
    profile = await trainer.train(
        texts=["角色对话记录.txt"],
        name_hint="傲娇少女_小夜",
    )
    # profile 可直接用于 create_from_preset()
"""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from alice.training.extractor import (
    CharacterExtractor, CharacterProfile, ENGINE_PARAMS_SPEC, TRAIT_PARAM_MAPPING
)
from alice.core.agent import LLMCall
from alice.training.styles import (
    StyleStore, StyleExample, StyleFingerprint,
    extract_style_fingerprint, merge_fingerprints,
)
from alice.training.rag import PersonalityRAG
from alice.storage.checkpoint import CheckpointManager, CheckpointMeta


# ------------------------------------------------------------
# 对话解析器（从文本中提取 user→character 对话对）
# ------------------------------------------------------------
class DialogueParser:
    """从自由文本中解析对话"""

    # 常见对话模式: "角色名: 对话内容"
    DIALOGUE_PATTERN = re.compile(
        r'(?:^|\n)\s*([^\s:：]{1,10})[：:]\s*(.+?)(?=\n[^\s:：]{1,10}[：:]|\n\n|\Z)',
        re.MULTILINE | re.DOTALL
    )

    # 日语/中文引号对话: 「对话」
    QUOTE_PATTERN = re.compile(r'[「『]([^」』]*)[」』]')

    # 常见叙述分隔
    NARRATION_PATTERN = re.compile(r'[（(][^)）]{2,30}[)）]')

    def __init__(self, character_names: Optional[List[str]] = None):
        """
        Args:
            character_names: 已知的角色名列表（用于识别哪句话是目标角色说的）
                            如果为 None，尝试自动识别
        """
        self.character_names = set(character_names or [])

    def parse(self, text: str, target_character: str = "") -> List[Tuple[str, str, str]]:
        """
        解析文本中的对话对。

        Returns:
            List of (speaker, content, context)
            context 是对话前的叙述文本或上一句对方说的话
        """
        dialogues = []

        # 方法 1: 命名对话格式 "A: ..."
        named_matches = self.DIALOGUE_PATTERN.findall(text)
        if named_matches:
            prev_speaker = ""
            prev_content = ""
            for speaker, content in named_matches:
                content = content.strip()
                if not content:
                    continue
                context = ""
                if prev_speaker and prev_speaker != speaker:
                    context = prev_content
                dialogues.append((speaker.strip(), content, context))
                prev_speaker = speaker.strip()
                prev_content = content
            return dialogues

        # 方法 2: 引号对话提取
        quotes = self.QUOTE_PATTERN.findall(text)
        if quotes:
            # 尝试根据上下文推断说话者
            # 简化处理: 奇数序号的引号是一个人，偶数是另一个人
            for i in range(len(quotes) - 1):
                content = quotes[i+1].strip()
                context = quotes[i].strip()
                speaker = target_character or ("角色" if i % 2 == 1 else "对方")
                dialogues.append((speaker, content, context))
            return dialogues

        # 方法 3: 按段落分割，每个段落作为一个对话片段
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        for para in paragraphs[:100]:
            lines = para.split('\n')
            if len(lines) >= 2:
                dialogues.append(("角色", lines[-1], lines[0] if len(lines) > 1 else ""))
            else:
                dialogues.append(("角色", para, ""))

        return dialogues


# ------------------------------------------------------------
# 对话采样器（LLM 驱动 — 提取最佳 few-shot 示例）
# ------------------------------------------------------------
class DialogueSampler:
    """从大量对话中筛选最能体现角色风格的示例"""

    SAMPLING_PROMPT = """你是一位专业的角色对话编辑。你的任务是从给定的对话记录中，挑选出最能体现角色说话风格的对话片段。

## 挑选标准
1. **风格典型性**: 该片段是否展示了角色独特的句式、用词、语气？
2. **情感表现力**: 该片段是否展示了角色的情绪反应方式？
3. **多样性**: 覆盖不同情境（开心、生气、关心、日常等）
4. **简洁性**: 对话不宜过长，控制在 50 字以内最佳

## 输出格式
返回 JSON 数组，每项包含:
```json
[
  {
    "context": "对方说了什么或当前场景",
    "character_response": "角色的回复",
    "tags": ["标签1", "标签2"],
    "quality_score": 0.9
  }
]
```

## 标签参考
使用以下标签分类: 傲娇, 温柔, 冷淡, 关心, 生气, 害羞, 日常, 撒娇, 嘴硬, 坦率, 幽默, 认真, 鼓励, 拒绝

请挑选 5-10 个最佳示例。"""

    def __init__(self, llm_call: Optional[LLMCall] = None):
        self.llm_call = llm_call

    async def sample(
        self, dialogues: List[Tuple[str, str, str]],
        target_character: str = "",
        max_samples: int = 10,
    ) -> List[Dict]:
        """
        从对话列表中采样最佳示例。

        Args:
            dialogues: DialogueParser.parse() 的输出
            target_character: 目标角色名
            max_samples: 最大采样数

        Returns:
            [{"context": ..., "character_response": ..., "tags": [...], "quality_score": ...}, ...]
        """
        # 过滤出目标角色的对话
        char_dialogues = []
        for speaker, content, context in dialogues:
            if not target_character or speaker == target_character or speaker == "角色":
                char_dialogues.append((content, context))

        if not char_dialogues:
            return []

        # 如果没有 LLM，用启发式方法采样
        if not self.llm_call:
            return self._heuristic_sample(char_dialogues, max_samples)

        # 用 LLM 采样
        dialogue_text = "\n".join(
            f"[{i}] 场景: {ctx}\n    角色回复: {resp}"
            for i, (resp, ctx) in enumerate(char_dialogues[:50])
        )

        try:
            response = await self.llm_call(
                self.SAMPLING_PROMPT,
                [{"role": "user", "content": f"目标角色: {target_character or '角色'}\n\n对话记录:\n{dialogue_text}"}]
            )

            # 解析 JSON
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                samples = json.loads(json_match.group(0))
                for s in samples:
                    s.setdefault("quality_score", 0.7)
                    s.setdefault("tags", [])
                return samples[:max_samples]
        except Exception:
            pass

        return self._heuristic_sample(char_dialogues, max_samples)

    def _heuristic_sample(self, dialogues: List[Tuple[str, str]], n: int) -> List[Dict]:
        """启发式采样: 按长度和多样性选"""
        # 按长度过滤（太短或太长的去掉）
        valid = [(resp, ctx) for resp, ctx in dialogues
                 if 10 <= len(resp) <= 100 and len(ctx) >= 2]

        if not valid:
            valid = dialogues

        # 均匀采样
        step = max(1, len(valid) // n)
        samples = []
        for i in range(0, len(valid), step):
            if len(samples) >= n:
                break
            resp, ctx = valid[i]
            samples.append({
                "context": ctx[:100],
                "character_response": resp,
                "tags": [],
                "quality_score": 0.5,
            })

        return samples


# ------------------------------------------------------------
# 风格提取器（LLM 驱动 — 深度分析角色说话方式）
# ------------------------------------------------------------
class StyleExtractor:
    """用 LLM 深度分析角色的语言风格"""

    STYLE_PROMPT = """你是一位专业的语言风格分析师。请根据提供的角色对话文本，深度分析该角色的说话风格。

## 分析维度

### 1. 句式特征
- 偏好什么句式？（短句/长句/反问句/感叹句/省略句）
- 句子节奏如何？（急促/平缓/跳跃）
- 句式变化是否丰富？

### 2. 词汇偏好
- 高频使用的独特词汇或短语
- 语气词使用习惯（啊/呀/呢/吧/嘛/哦 等的偏好）
- 是否有独特的口头禅或语言标记

### 3. 修辞手法
- 常用什么修辞？（比喻/夸张/反问/设问/排比）
- 说话的"腔调"如何？（俏皮/严肃/慵懒/活泼/优雅/粗鲁）

### 4. 交互模式
- 如何回应对方的关心？（接受/拒绝/嘴硬/感激）
- 如何回应对方的批评？（防御/接受/反击/沉默）
- 主动说话的频率和方式

### 5. 情绪表达
- 情绪流露方式（直接/含蓄/夸张/克制）
- 不同情绪下的语言变化（开心时怎样、生气时怎样、难过时怎样）

## 输出格式
返回 JSON:
```json
{
  "sentence_patterns": "句式特征描述",
  "vocabulary_style": "词汇偏好描述",
  "rhetoric_style": "修辞和腔调描述",
  "interaction_style": "交互模式描述",
  "emotion_expression": "情绪表达方式描述",
  "overall_style_summary": "2-3句话的总体风格总结"
}
```"""

    def __init__(self, llm_call: Optional[LLMCall] = None):
        self.llm_call = llm_call

    async def extract(self, character_texts: List[str]) -> Dict[str, str]:
        """
        深度提取角色风格。

        Args:
            character_texts: 角色的对话文本列表

        Returns:
            风格分析结果字典
        """
        if not character_texts:
            return {}

        text = "\n".join(character_texts[:100])  # 最多取 100 条

        if not self.llm_call:
            return {}

        try:
            response = await self.llm_call(
                self.STYLE_PROMPT,
                [{"role": "user", "content": f"角色对话文本:\n{text[:6000]}"}]
            )

            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception:
            pass

        return {}


# ------------------------------------------------------------
# PersonalityTrainer — 主训练器
# ------------------------------------------------------------
class PersonalityTrainer:
    """
    人格训练器：多阶段管道，将文本转化为角色人格。

    用法:
        trainer = PersonalityTrainer(llm_call=my_llm_func)

        # 首次训练
        profile = await trainer.train(
            texts=["角色对话.txt"],
            name_hint="傲娇少女_小夜",
        )

        # 增量训练
        profile = await trainer.train_incremental(
            preset_name="傲娇少女_小夜",
            texts=["新找到的对话.txt"],
        )
    """

    def __init__(
        self,
        llm_call: Optional[LLMCall] = None,
        style_store: Optional[StyleStore] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        embed_fn=None,  # async (texts: List[str]) -> List[List[float]]
    ):
        self.llm_call = llm_call
        self.embed_fn = embed_fn
        self.style_store = style_store or StyleStore(embed_fn=embed_fn)
        self.checkpoint = checkpoint_manager or CheckpointManager()

        # 子模块
        self.character_extractor = CharacterExtractor()
        self.dialogue_parser = DialogueParser()
        self.dialogue_sampler = DialogueSampler(llm_call)
        self.style_extractor = StyleExtractor(llm_call)
        self.personality_rag = PersonalityRAG(self.style_store)

    # ----------------------------------------------------------------
    # Phase 1: 语料预处理
    # ----------------------------------------------------------------
    def preprocess_corpus(
        self, texts: List[str], file_paths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        预处理输入文本。

        Returns:
            {
                "full_text": str,              # 合并后的全量文本
                "paragraphs": List[str],        # 段落列表
                "character_texts": List[str],   # 角色说的话
                "dialogue_pairs": List[Tuple],  # (speaker, content, context)
                "total_chars": int,
            }
        """
        # 读取文件
        all_text_parts = []
        all_files = file_paths or []

        if texts:
            for i, text in enumerate(texts):
                if text.strip():
                    all_text_parts.append(text.strip())
                    # 如果文本看起来像文件路径且存在
                    path = Path(text.strip())
                    if path.exists() and path.is_file():
                        all_files.append(str(path))
                        all_text_parts[-1] = path.read_text(encoding="utf-8")

        if all_files:
            for fp in all_files:
                try:
                    content = Path(fp).read_text(encoding="utf-8")
                    all_text_parts.append(f"--- {Path(fp).name} ---\n{content}")
                except Exception:
                    pass

        full_text = "\n\n".join(all_text_parts)

        # 分段
        paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]

        # 解析对话
        dialogues = self.dialogue_parser.parse(full_text)

        # 提取角色说的话
        character_texts = []
        for speaker, content, _ in dialogues:
            if speaker not in ('旁白', '叙述', 'narrator'):
                character_texts.append(content)

        return {
            "full_text": full_text,
            "paragraphs": paragraphs,
            "character_texts": character_texts,
            "dialogue_pairs": dialogues,
            "total_chars": len(full_text),
        }

    # ----------------------------------------------------------------
    # Phase 2: 多维度提取
    # ----------------------------------------------------------------
    async def extract_features(
        self,
        preprocessed: Dict[str, Any],
        name_hint: str = "",
        preset_name: str = "",
    ) -> Dict[str, Any]:
        """
        多维度特征提取。

        Returns:
            {
                "profile": CharacterProfile,       # 结构化画像（兼容现有格式）
                "fingerprint": StyleFingerprint,   # 统计风格指纹
                "style_analysis": Dict,            # LLM 深度风格分析
                "examples": List[StyleExample],    # 对话示例
                "batch_id": str,                   # 批次 ID
            }
        """
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        full_text = preprocessed["full_text"]
        character_texts = preprocessed["character_texts"]
        dialogues = preprocessed["dialogue_pairs"]

        # ---- 维度 1: 结构化画像（复用现有 CharacterExtractor） ----
        profile = None
        if self.llm_call:
            try:
                profile = await self.character_extractor.extract(
                    llm_call=self.llm_call,
                    text=full_text[:12000],  # 保留限制但只用头部
                    name_hint=name_hint,
                )
            except Exception as e:
                print(f"[Trainer] 结构化画像提取失败: {e}")

        if profile is None:
            profile = CharacterProfile(
                name=name_hint or preset_name or "未命名角色",
                source_summary="从文本中提取",
            )

        # ---- 维度 2: 统计风格指纹（非 LLM） ----
        fingerprint = extract_style_fingerprint(
            texts=character_texts if character_texts else [full_text],
            preset_name=preset_name or profile.name,
        )

        # ---- 维度 3: LLM 深度风格分析 ----
        style_analysis = {}
        if self.llm_call and character_texts:
            try:
                style_analysis = await self.style_extractor.extract(character_texts)
                # 将 LLM 风格分析结果融入 profile
                if style_analysis:
                    self._merge_style_into_profile(profile, style_analysis)
            except Exception as e:
                print(f"[Trainer] 风格分析失败: {e}")

        # ---- 维度 4: 对话示例采样 ----
        samples = await self.dialogue_sampler.sample(
            dialogues=dialogues,
            target_character=name_hint or preset_name,
            max_samples=10,
        )

        # 转换为 StyleExample
        examples = []
        for s in samples:
            examples.append(StyleExample(
                preset_name=preset_name or profile.name,
                source_file=",".join(preprocessed.get("file_paths", ["直接输入"])),
                batch_id=batch_id,
                context=s.get("context", ""),
                character_response=s.get("character_response", ""),
                tags=s.get("tags", []),
                quality_score=s.get("quality_score", 0.5),
            ))

        # ---- 维度 5: 增强 conversation_examples ----
        if examples and not profile.conversation_examples:
            profile.conversation_examples = [
                {"user": ex.context, "character": ex.character_response}
                for ex in examples[:4]
            ]

        return {
            "profile": profile,
            "fingerprint": fingerprint,
            "style_analysis": style_analysis,
            "examples": examples,
            "batch_id": batch_id,
        }

    def _merge_style_into_profile(self, profile: CharacterProfile, style: Dict):
        """将 LLM 风格分析融入 profile"""
        cp = profile.character_profile
        spb = profile.system_prompt_blocks

        # 增强 speech_style
        existing_speech = cp.get("speech_style", "")
        parts = [existing_speech] if existing_speech else []

        if style.get("sentence_patterns"):
            parts.append(f"句式: {style['sentence_patterns']}")
        if style.get("vocabulary_style"):
            parts.append(f"用词: {style['vocabulary_style']}")
        if style.get("rhetoric_style"):
            parts.append(f"腔调: {style['rhetoric_style']}")

        if parts:
            cp["speech_style"] = "；".join(parts)

        # 增强 behavioral_patterns
        existing_behavior = cp.get("behavioral_patterns", "")
        behavior_parts = [existing_behavior] if existing_behavior else []

        if style.get("interaction_style"):
            behavior_parts.append(f"互动: {style['interaction_style']}")
        if style.get("emotion_expression"):
            behavior_parts.append(f"情绪表达: {style['emotion_expression']}")

        if behavior_parts:
            cp["behavioral_patterns"] = "；".join(behavior_parts)

        # 增强 speech_rules
        if style.get("overall_style_summary") and not spb.get("speech_rules"):
            spb["speech_rules"] = f"- {style['overall_style_summary']}"

    # ----------------------------------------------------------------
    # Phase 3: 向量索引
    # ----------------------------------------------------------------
    def index_features(self, preset_name: str, examples: List[StyleExample],
                       fingerprint: StyleFingerprint, corpus_text: str = ""):
        """将提取的特征存入向量存储和指纹存储"""
        if examples:
            for ex in examples:
                ex.preset_name = preset_name
            self.style_store.add_examples_batch(examples)

        if fingerprint:
            fingerprint.preset_name = preset_name
            self.style_store.save_fingerprint(fingerprint)

    # ----------------------------------------------------------------
    # Phase 4: 增量融合
    # ----------------------------------------------------------------
    async def incremental_fusion(
        self, preset_name: str, new_features: Dict[str, Any]
    ) -> CharacterProfile:
        """
        将新提取的特征与已有特征融合。

        新数据权重 = max(0.3, new_chars / (old_chars + new_chars))
        更多数据 = 更高权重。
        """
        old_profile = self.checkpoint.load_profile(preset_name)
        old_fingerprint = self.style_store.load_fingerprint(preset_name)
        old_examples_count = self.style_store.get_example_count(preset_name)

        new_profile = new_features["profile"]
        new_fingerprint = new_features["fingerprint"]
        new_examples = new_features["examples"]

        # ---- 融合指纹 ----
        if old_fingerprint and new_fingerprint:
            old_chars = old_fingerprint.total_chars_analyzed
            new_chars = new_fingerprint.total_chars_analyzed
            old_weight = old_chars / (old_chars + new_chars) if (old_chars + new_chars) > 0 else 0.5
            merged_fingerprint = merge_fingerprints(old_fingerprint, new_fingerprint, old_weight)
            self.style_store.save_fingerprint(merged_fingerprint)
        elif new_fingerprint:
            self.style_store.save_fingerprint(new_fingerprint)

        # ---- 融合 profile ----
        if old_profile:
            # 保留旧的 conversation_examples 中高质量的部分
            old_examples = old_profile.conversation_examples or []
            new_examples_raw = new_profile.conversation_examples or []
            # 合并去重（按 character 回复的前 20 字去重）
            seen = set()
            merged_conv_examples = []
            for ex in new_examples_raw + old_examples:
                key = ex.get("character", "")[:20]
                if key and key not in seen:
                    seen.add(key)
                    merged_conv_examples.append(ex)
            new_profile.conversation_examples = merged_conv_examples[:8]

            # 如果新 profile 的 name 更准确，保留
            if old_profile.name and not new_profile.name.startswith("未命名"):
                pass  # 使用新的 name

            # rates 做加权平均（旧:新 = old_weight:new_weight）
            old_rates = old_profile.rates or {}
            new_rates = new_profile.rates or {}
            old_chars = old_fingerprint.total_chars_analyzed if old_fingerprint else 0
            new_chars = new_fingerprint.total_chars_analyzed if new_fingerprint else 1
            old_w = old_chars / (old_chars + new_chars) if (old_chars + new_chars) > 0 else 0.5
            new_w = 1.0 - old_w

            for key in set(list(old_rates.keys()) + list(new_rates.keys())):
                old_val = old_rates.get(key, 0)
                new_val = new_rates.get(key, 0)
                if old_val and new_val:
                    new_profile.rates[key] = old_val * old_w + new_val * new_w
                elif new_val:
                    new_profile.rates[key] = new_val
                # else keep old_val (already in new_profile.rates via merge)

            merged_profile = new_profile
        else:
            merged_profile = new_profile

        # ---- 索引新示例 ----
        if new_examples:
            self.index_features(preset_name, new_examples, new_fingerprint)

        return merged_profile

    # ----------------------------------------------------------------
    # 主入口: 首次训练
    # ----------------------------------------------------------------
    async def train(
        self,
        texts: Optional[List[str]] = None,
        files: Optional[List[str]] = None,
        name_hint: str = "",
    ) -> CharacterProfile:
        """
        首次训练: 从零开始构建角色人格。

        Args:
            texts: 直接输入的文本列表
            files: 文件路径列表
            name_hint: 角色名提示

        Returns:
            CharacterProfile — 可直接用于 create_from_preset()
        """
        print(f"[Trainer] ===== 开始训练: {name_hint or '未命名'} =====")

        # Phase 1: 预处理
        print("[Trainer] Phase 1: 语料预处理...")
        preprocessed = self.preprocess_corpus(texts or [], files or [])
        total_chars = preprocessed["total_chars"]
        print(f"[Trainer]   总字符数: {total_chars}, 对话数: {len(preprocessed['dialogue_pairs'])}")

        # Phase 2: 多维度提取
        print("[Trainer] Phase 2: 多维度特征提取...")
        features = await self.extract_features(
            preprocessed, name_hint=name_hint,
        )
        profile = features["profile"]
        print(f"[Trainer]   角色: {profile.name}, 示例数: {len(features['examples'])}")

        # Phase 3: 向量索引
        print("[Trainer] Phase 3: 向量索引...")
        preset_name = profile.name
        self.index_features(
            preset_name=preset_name,
            examples=features["examples"],
            fingerprint=features["fingerprint"],
        )

        # 保存语料
        source_files = files or []
        if texts and not source_files:
            source_files = ["直接输入"]

        for sf in source_files:
            try:
                content = Path(sf).read_text(encoding="utf-8") if Path(sf).exists() else sf
                self.checkpoint.save_corpus(
                    preset_name=preset_name,
                    text=content if Path(sf).exists() else "\n".join(texts or []),
                    source_name=Path(sf).name if Path(sf).exists() else "直接输入",
                    batch_id=features["batch_id"],
                )
            except Exception:
                pass

        # Phase 5: 保存存档
        print("[Trainer] Phase 5: 保存存档...")
        meta = CheckpointMeta(
            preset_name=preset_name,
            total_corpus_chars=total_chars,
            total_corpus_files=len(source_files) if source_files else 1,
            total_batches=1,
            total_examples=len(features["examples"]),
            last_training_at=datetime.now().isoformat(),
        )
        self.checkpoint.save_profile(profile)
        self.checkpoint.save_checkpoint_meta(meta)

        print(f"[Trainer] ===== 训练完成: {preset_name} =====")
        return profile

    # ----------------------------------------------------------------
    # 增量训练
    # ----------------------------------------------------------------
    async def train_incremental(
        self,
        preset_name: str,
        texts: Optional[List[str]] = None,
        files: Optional[List[str]] = None,
    ) -> CharacterProfile:
        """
        增量训练: 在已有角色的基础上添加新文本，校准人格。

        Args:
            preset_name: 已有的角色预设名
            texts: 新增文本列表
            files: 新增文件路径列表

        Returns:
            更新后的 CharacterProfile
        """
        print(f"[Trainer] ===== 增量训练: {preset_name} =====")

        # 加载已有存档
        old_meta = self.checkpoint.load_checkpoint_meta(preset_name)
        old_profile = self.checkpoint.load_profile(preset_name)

        # Phase 1: 预处理新文本
        print("[Trainer] Phase 1: 预处理新语料...")
        preprocessed = self.preprocess_corpus(texts or [], files or [])
        new_chars = preprocessed["total_chars"]
        print(f"[Trainer]   新增字符数: {new_chars}")

        # Phase 2: 提取新特征
        print("[Trainer] Phase 2: 提取新特征...")
        features = await self.extract_features(
            preprocessed,
            name_hint=preset_name,
            preset_name=preset_name,
        )

        # Phase 4: 增量融合
        print("[Trainer] Phase 4: 增量融合...")
        merged_profile = await self.incremental_fusion(preset_name, features)

        # 保存新语料
        source_files = files or []
        if texts and not source_files:
            source_files = ["直接输入"]
        for sf in source_files:
            try:
                content = Path(sf).read_text(encoding="utf-8") if Path(sf).exists() else sf
                self.checkpoint.save_corpus(
                    preset_name=preset_name,
                    text=content if Path(sf).exists() else "\n".join(texts or []),
                    source_name=Path(sf).name if Path(sf).exists() else "直接输入",
                    batch_id=features["batch_id"],
                )
            except Exception:
                pass

        # 更新 meta
        old_meta.total_corpus_chars += new_chars
        old_meta.total_corpus_files += len(source_files) if source_files else 1
        old_meta.total_batches += 1
        old_meta.total_examples = self.style_store.get_example_count(preset_name)
        old_meta.last_training_at = datetime.now().isoformat()
        old_meta.version += 1

        self.checkpoint.save_profile(merged_profile)
        self.checkpoint.save_checkpoint_meta(old_meta)

        print(f"[Trainer] ===== 增量训练完成: {preset_name} (v{old_meta.version}) =====")
        return merged_profile

    # ----------------------------------------------------------------
    # 使用反馈校准（RLHF 类比）
    # ----------------------------------------------------------------
    async def incorporate_feedback(self, preset_name: str) -> bool:
        """
        将用户纠正记录融入风格示例库。

        用户纠正的"正确回复"作为正例加入 style_store，
        被纠正的"原始回复"降低 quality_score。
        """
        feedback_records = self.checkpoint.get_feedback_examples(preset_name)
        if not feedback_records:
            return False

        new_examples = []
        for record in feedback_records:
            # 将纠正后的回复作为高质量示例
            new_examples.append(StyleExample(
                preset_name=preset_name,
                source_file="feedback",
                batch_id="feedback_correction",
                context=record.get("user_message", ""),
                character_response=record.get("corrected_response", ""),
                tags=["用户纠正"],
                quality_score=0.9,  # 用户纠正的示例质量最高
            ))

        if new_examples:
            self.style_store.add_examples_batch(new_examples)
            print(f"[Trainer] 已融入 {len(new_examples)} 条用户反馈")
            return True

        return False

    # ----------------------------------------------------------------
    # 便捷方法
    # ----------------------------------------------------------------
    def get_checkpoint_summary(self, preset_name: str) -> Dict:
        """获取训练摘要"""
        return self.checkpoint.get_checkpoint_summary(preset_name)

    def list_presets(self) -> List[str]:
        """列出所有可用预设"""
        from alice.storage.preset import PresetManager
        return PresetManager().list_all()
