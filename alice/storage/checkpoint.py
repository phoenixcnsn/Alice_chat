"""
人格存档管理器 — 完整 Checkpoint 系统

像 Galgame 存档一样保存角色的全部状态。
类似于训练大模型时的 checkpoint 保存。

存档目录结构:
  presets/<name>/
  ├── checkpoint.json        # 元数据: 版本、时间戳、训练统计
  ├── profile.json           # CharacterProfile (增强版)
  ├── engine_state.json      # 积温引擎运行时状态
  ├── style_index/           # ChromaDB 风格向量 (另由 style_store 管理)
  ├── corpus/                # 原始语料 (永久保留)
  │   ├── batch_001.txt
  │   └── ...
  ├── conversations/         # 对话日志
  │   └── convo_20260612.jsonl
  └── feedback/              # 用户纠正记录 (RLHF 数据)
      └── corrections.jsonl

与 preset_manager.py 的关系:
  - preset_manager: 管理 profile.json 的 CRUD
  - checkpoint_manager: 管理完整存档（profile + engine + corpus + conversations）
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from alice.training.extractor import CharacterProfile


# ------------------------------------------------------------
# 数据模型
# ------------------------------------------------------------
@dataclass
class CheckpointMeta:
    """存档元数据"""
    preset_name: str
    version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    # 训练统计
    total_corpus_chars: int = 0           # 累计训练的字符数
    total_corpus_files: int = 0           # 累计训练的文件数
    total_batches: int = 0                # 累计训练批次
    total_examples: int = 0               # 风格示例总数
    total_conversations: int = 0          # 对话轮数
    last_training_at: str = ""            # 最后一次训练时间
    # 引擎状态标记
    engine_state_saved: bool = False      # 是否保存了引擎状态
    engine_virtual_time_min: float = 0.0  # 引擎虚拟时间

    def to_dict(self) -> Dict:
        return {
            "preset_name": self.preset_name,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "total_corpus_chars": self.total_corpus_chars,
            "total_corpus_files": self.total_corpus_files,
            "total_batches": self.total_batches,
            "total_examples": self.total_examples,
            "total_conversations": self.total_conversations,
            "last_training_at": self.last_training_at,
            "engine_state_saved": self.engine_state_saved,
            "engine_virtual_time_min": self.engine_virtual_time_min,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "CheckpointMeta":
        return cls(
            preset_name=d.get("preset_name", ""),
            version=d.get("version", 1),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            total_corpus_chars=d.get("total_corpus_chars", 0),
            total_corpus_files=d.get("total_corpus_files", 0),
            total_batches=d.get("total_batches", 0),
            total_examples=d.get("total_examples", 0),
            total_conversations=d.get("total_conversations", 0),
            last_training_at=d.get("last_training_at", ""),
            engine_state_saved=d.get("engine_state_saved", False),
            engine_virtual_time_min=d.get("engine_virtual_time_min", 0.0),
        )


@dataclass
class ConversationLog:
    """单条对话日志"""
    role: str                               # "user" | "assistant" | "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    emotion_state: Optional[Dict] = None    # 当时的情绪状态快照
    feedback: Optional[str] = None          # 用户反馈 (good/bad/correct)

    def to_dict(self) -> Dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "emotion_state": self.emotion_state,
            "feedback": self.feedback,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ConversationLog":
        return cls(
            role=d.get("role", ""),
            content=d.get("content", ""),
            timestamp=d.get("timestamp", 0),
            emotion_state=d.get("emotion_state"),
            feedback=d.get("feedback"),
        )


# ------------------------------------------------------------
# CheckpointManager
# ------------------------------------------------------------
class CheckpointManager:
    """完整存档管理器"""

    def __init__(self, presets_dir: str = "presets"):
        self.presets_dir = Path(presets_dir)

    # ----------------------------------------------------------------
    # 路径工具
    # ----------------------------------------------------------------
    def get_preset_dir(self, preset_name: str) -> Path:
        """获取预设目录"""
        return self.presets_dir / preset_name

    def _ensure_dirs(self, preset_name: str):
        """确保预设的所有子目录存在"""
        base = self.get_preset_dir(preset_name)
        base.mkdir(parents=True, exist_ok=True)
        (base / "corpus").mkdir(exist_ok=True)
        (base / "conversations").mkdir(exist_ok=True)
        (base / "feedback").mkdir(exist_ok=True)
        return base

    # ----------------------------------------------------------------
    # Profile (委托给 preset_manager，这里做增强)
    # ----------------------------------------------------------------
    def save_profile(self, profile: CharacterProfile):
        """保存角色画像（兼容 preset_manager 的 JSON 格式）"""
        base = self._ensure_dirs(profile.name)
        filepath = base / "profile.json"
        profile.created_at = profile.created_at or datetime.now().isoformat()
        data = profile.to_dict()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_profile(self, preset_name: str) -> Optional[CharacterProfile]:
        """加载角色画像"""
        filepath = self.get_preset_dir(preset_name) / "profile.json"
        if not filepath.exists():
            # 回退到旧格式
            old_path = self.presets_dir / f"{preset_name}.json"
            if old_path.exists():
                return CharacterProfile.from_dict(
                    json.loads(old_path.read_text(encoding="utf-8"))
                )
            return None
        return CharacterProfile.from_dict(
            json.loads(filepath.read_text(encoding="utf-8"))
        )

    # ----------------------------------------------------------------
    # Checkpoint 元数据
    # ----------------------------------------------------------------
    def save_checkpoint_meta(self, meta: CheckpointMeta):
        """保存存档元数据"""
        base = self._ensure_dirs(meta.preset_name)
        meta.updated_at = datetime.now().isoformat()
        filepath = base / "checkpoint.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(meta.to_dict(), f, ensure_ascii=False, indent=2)

    def load_checkpoint_meta(self, preset_name: str) -> CheckpointMeta:
        """加载存档元数据"""
        filepath = self.get_preset_dir(preset_name) / "checkpoint.json"
        if filepath.exists():
            return CheckpointMeta.from_dict(
                json.loads(filepath.read_text(encoding="utf-8"))
            )
        return CheckpointMeta(preset_name=preset_name)

    # ----------------------------------------------------------------
    # 引擎状态持久化
    # ----------------------------------------------------------------
    async def save_engine_state(self, preset_name: str, agent) -> bool:
        """
        保存积温引擎的完整运行时状态。

        Args:
            preset_name: 角色预设名
            agent: PersonalityAgent 实例

        Returns:
            是否保存成功
        """
        if not agent or not agent.engine:
            return False

        try:
            state = await agent.get_state()
            diag = agent.get_diagnostics()

            # 对话历史
            chat_history = []
            if hasattr(agent, 'conversation_history'):
                for m in agent.conversation_history:
                    chat_history.append({
                        "role": m.role,
                        "content": m.content,
                        "timestamp": m.timestamp if hasattr(m, 'timestamp') else time.time(),
                    })

            engine_data = {
                "saved_at": datetime.now().isoformat(),
                "virtual_time": diag.get("virtual_time_min", 0.0),
                "connection": state.get("connection", 0.0),
                "pride": state.get("pride", 0.0),
                "valence": state.get("valence", 0.0),
                "valence_vel": agent.engine._valence_vel,
                "arousal": state.get("arousal", 0.0),
                "immersion": state.get("immersion", 0.0),
                "net_willingness": state.get("net_willingness", 0.0),
                "zone": diag.get("zone", "idle"),
                "mood_quadrant": diag.get("mood_quadrant", "neutral"),
                "last_activity": str(agent.engine._last_activity) if agent.engine._last_activity else None,
                "chat_history": chat_history,  # 对话历史
                # 保存完整的引擎参数以便恢复
                "rates": {
                    k: getattr(agent.engine, k)
                    for k in dir(agent.engine)
                    if k.startswith(('conn_', 'pride_', 'valence_', 'arousal_',
                                     'immersion_', 'activity_', 'kappa'))
                    and not k.startswith('_')
                },
                "thresholds": dict(agent.engine.thresholds) if hasattr(agent.engine, 'thresholds') else {},
                "persona": dict(agent.engine.persona) if hasattr(agent.engine, 'persona') else {},
            }

            base = self._ensure_dirs(preset_name)
            filepath = base / "engine_state.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(engine_data, f, ensure_ascii=False, indent=2)

            # 更新 checkpoint meta
            meta = self.load_checkpoint_meta(preset_name)
            meta.engine_state_saved = True
            meta.engine_virtual_time_min = diag.get("virtual_time_min", 0.0)
            self.save_checkpoint_meta(meta)

            return True
        except Exception as e:
            print(f"[CheckpointManager] 保存引擎状态失败: {e}")
            return False

    def load_engine_state(self, preset_name: str) -> Optional[Dict]:
        """加载引擎状态"""
        filepath = self.get_preset_dir(preset_name) / "engine_state.json"
        if filepath.exists():
            return json.loads(filepath.read_text(encoding="utf-8"))
        return None

    # ----------------------------------------------------------------
    # 语料管理（源文本永久保留）
    # ----------------------------------------------------------------
    def save_corpus(self, preset_name: str, text: str,
                    source_name: str = "", batch_id: str = "") -> Path:
        """
        保存一份源文本到 corpus/ 目录。

        Args:
            preset_name: 角色预设名
            text: 源文本内容
            source_name: 原始文件名或描述
            batch_id: 批次 ID

        Returns:
            保存的文件路径
        """
        base = self._ensure_dirs(preset_name)
        corpus_dir = base / "corpus"

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = source_name.replace("/", "_").replace("\\", "_") if source_name else "input"
        if batch_id:
            filename = f"{timestamp}_{batch_id}_{safe_name}.txt"
        else:
            filename = f"{timestamp}_{safe_name}.txt"

        # 确保唯一
        filepath = corpus_dir / filename
        if filepath.exists():
            stem = filepath.stem
            filepath = corpus_dir / f"{stem}_{int(time.time()*1000)%10000}.txt"

        filepath.write_text(text, encoding="utf-8")
        print(f"[CheckpointManager] 语料已保存: {filepath}")
        return filepath

    def list_corpus(self, preset_name: str) -> List[Path]:
        """列出所有语料文件"""
        corpus_dir = self.get_preset_dir(preset_name) / "corpus"
        if not corpus_dir.exists():
            return []
        return sorted(corpus_dir.glob("*.txt"))

    def read_all_corpus(self, preset_name: str) -> str:
        """读取该预设的全部语料（合并）"""
        files = self.list_corpus(preset_name)
        texts = []
        for fp in files:
            try:
                texts.append(f"--- {fp.name} ---\n{fp.read_text(encoding='utf-8')}")
            except Exception:
                texts.append(f"[读取失败: {fp.name}]")
        return "\n\n".join(texts)

    # ----------------------------------------------------------------
    # 对话日志
    # ----------------------------------------------------------------
    def log_conversation(self, preset_name: str, log: ConversationLog):
        """追加一条对话日志"""
        base = self._ensure_dirs(preset_name)
        today = datetime.now().strftime("%Y%m%d")
        logfile = base / "conversations" / f"convo_{today}.jsonl"

        with open(logfile, "a", encoding="utf-8") as f:
            f.write(json.dumps(log.to_dict(), ensure_ascii=False) + "\n")

    def get_conversation_logs(self, preset_name: str, days: int = 7) -> List[ConversationLog]:
        """获取最近 N 天的对话日志"""
        base = self.get_preset_dir(preset_name) / "conversations"
        if not base.exists():
            return []

        logs = []
        for logfile in sorted(base.glob("convo_*.jsonl"), reverse=True):
            try:
                for line in logfile.read_text(encoding="utf-8").strip().split("\n"):
                    if line:
                        logs.append(ConversationLog.from_dict(json.loads(line)))
            except Exception:
                pass
            if len(logs) > 10000:
                break

        # 按时间过滤
        cutoff = time.time() - days * 86400
        return [l for l in logs if l.timestamp >= cutoff]

    def count_conversations(self, preset_name: str) -> int:
        """统计对话轮数"""
        base = self.get_preset_dir(preset_name) / "conversations"
        if not base.exists():
            return 0
        count = 0
        for logfile in base.glob("convo_*.jsonl"):
            try:
                count += len(logfile.read_text(encoding="utf-8").strip().split("\n"))
            except Exception:
                pass
        return count

    # ----------------------------------------------------------------
    # 用户反馈 (RLHF 数据)
    # ----------------------------------------------------------------
    def save_feedback(self, preset_name: str, user_message: str,
                      original_response: str, corrected_response: str,
                      reason: str = ""):
        """
        保存用户对角色回复的纠正。

        Args:
            preset_name: 角色预设名
            user_message: 用户当时说的话
            original_response: 角色原始回复
            corrected_response: 用户期望的回复
            reason: 纠正原因
        """
        base = self._ensure_dirs(preset_name)
        feedback_file = base / "feedback" / "corrections.jsonl"

        record = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "original_response": original_response,
            "corrected_response": corrected_response,
            "reason": reason,
        }

        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[CheckpointManager] 反馈已记录: {preset_name}")

    def get_feedback_examples(self, preset_name: str, limit: int = 20) -> List[Dict]:
        """获取用户反馈示例（可用于反向训练）"""
        feedback_file = self.get_preset_dir(preset_name) / "feedback" / "corrections.jsonl"
        if not feedback_file.exists():
            return []

        records = []
        try:
            for line in feedback_file.read_text(encoding="utf-8").strip().split("\n"):
                if line:
                    records.append(json.loads(line))
        except Exception:
            pass

        return records[-limit:]

    # ----------------------------------------------------------------
    # 完整存档（一键保存/加载全部状态）
    # ----------------------------------------------------------------
    async def save_full_checkpoint(
        self,
        profile: CharacterProfile,
        agent=None,                    # PersonalityAgent 实例
        meta: Optional[CheckpointMeta] = None,
    ) -> bool:
        """
        一键保存完整存档：profile + engine state + checkpoint meta。

        Args:
            profile: 角色画像
            agent: PersonalityAgent 实例（可选，有则保存引擎状态）
            meta: 存档元数据（可选）

        Returns:
            是否全部保存成功
        """
        preset_name = profile.name

        # 1. 保存 profile
        self.save_profile(profile)

        # 2. 保存引擎状态
        engine_ok = True
        if agent:
            engine_ok = await self.save_engine_state(preset_name, agent)

        # 3. 保存 meta
        if meta is None:
            meta = self.load_checkpoint_meta(preset_name)
        self.save_checkpoint_meta(meta)

        print(f"[CheckpointManager] 完整存档已保存: {preset_name}")
        return engine_ok

    async def load_full_checkpoint(
        self, preset_name: str, agent=None
    ) -> Dict[str, Any]:
        """
        一键加载完整存档。

        Returns:
            {
                "profile": CharacterProfile or None,
                "engine_state": Dict or None,
                "meta": CheckpointMeta,
                "corpus_files": List[Path],
            }
        """
        result = {
            "profile": self.load_profile(preset_name),
            "engine_state": self.load_engine_state(preset_name),
            "meta": self.load_checkpoint_meta(preset_name),
            "corpus_files": self.list_corpus(preset_name),
        }

        # 如果有 agent，恢复引擎状态
        if agent and result["engine_state"]:
            es = result["engine_state"]
            try:
                # 恢复积温引擎的数值状态
                if hasattr(agent.engine, '_connection'):
                    agent.engine._connection = es.get("connection", 0.0)
                if hasattr(agent.engine, '_pride'):
                    agent.engine._pride = es.get("pride", 0.0)
                if hasattr(agent.engine, '_valence'):
                    agent.engine._valence = es.get("valence", 0.0)
                if hasattr(agent.engine, '_arousal'):
                    agent.engine._arousal = es.get("arousal", 0.0)
                if hasattr(agent.engine, '_immersion'):
                    agent.engine._immersion = es.get("immersion", 0.0)
                if hasattr(agent.engine, '_virtual_time'):
                    agent.engine._virtual_time = es.get("virtual_time", 0.0)
                print(f"[CheckpointManager] 引擎状态已恢复: {preset_name}")
            except Exception as e:
                print(f"[CheckpointManager] 恢复引擎状态失败: {e}")

        return result

    # ----------------------------------------------------------------
    # 预设删除（完整清理）
    # ----------------------------------------------------------------
    def delete_preset_full(self, preset_name: str) -> bool:
        """
        完整删除预设（包括所有存档文件）。

        警告: 不可逆！会删除 corpus/ 下所有原始语料。
        """
        if preset_name == "默认":
            print("[CheckpointManager] 不允许删除默认预设")
            return False

        preset_dir = self.get_preset_dir(preset_name)
        if preset_dir.exists():
            shutil.rmtree(preset_dir)
            print(f"[CheckpointManager] 已完全删除: {preset_dir}")
            return True

        # 兼容旧格式
        old_path = self.presets_dir / f"{preset_name}.json"
        if old_path.exists():
            old_path.unlink()
            print(f"[CheckpointManager] 已删除旧格式预设: {old_path}")
            return True

        return False

    # ----------------------------------------------------------------
    # 存档列表（用于 UI 展示存档状态）
    # ----------------------------------------------------------------
    def get_checkpoint_summary(self, preset_name: str) -> Dict:
        """获取存档摘要（用于 UI 展示）"""
        meta = self.load_checkpoint_meta(preset_name)
        corpus_files = self.list_corpus(preset_name)
        convo_count = self.count_conversations(preset_name)

        return {
            "preset_name": preset_name,
            "version": meta.version,
            "created_at": meta.created_at,
            "updated_at": meta.updated_at,
            "corpus_files": len(corpus_files),
            "corpus_chars": meta.total_corpus_chars,
            "style_examples": meta.total_examples,
            "conversations": convo_count,
            "engine_saved": meta.engine_state_saved,
            "virtual_time_min": meta.engine_virtual_time_min,
            "has_feedback": (self.get_preset_dir(preset_name) / "feedback" / "corrections.jsonl").exists(),
        }
