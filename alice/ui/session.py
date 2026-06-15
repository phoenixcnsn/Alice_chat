"""
AgentSession — 管理 PersonalityAgent 的生命周期，支持 LLM 热切换和角色预设。
"""

import asyncio
import time
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field

from alice.core.agent import (
    PersonalityAgent, create_from_preset, ConversationMessage,
    create_anthropic_adapter, create_openai_adapter, create_deepseek_adapter,
)
from alice.storage.preset import PresetManager
from alice.storage.checkpoint import CheckpointManager
from alice.training.styles import StyleStore
from alice.training.trainer import PersonalityTrainer
from alice.training.extractor import CharacterProfile


@dataclass
class AgentSession:
    """管理 PersonalityAgent 的完整生命周期"""

    presets_dir: str = "presets"

    # ---- 组件 ----
    agent: Optional[PersonalityAgent] = None
    preset_name: str = "默认"
    llm_connected: bool = False
    llm_provider: str = "无"

    preset_manager: PresetManager = field(default_factory=lambda: PresetManager("presets"))
    checkpoint_manager: CheckpointManager = field(default_factory=lambda: CheckpointManager("presets"))
    style_store: StyleStore = field(default_factory=StyleStore)
    trainer: Optional[PersonalityTrainer] = None
    reference_manager: Any = field(default=None, repr=False)

    # ---- 状态缓存 ----
    last_state: Dict = field(default_factory=dict)
    last_diag: Dict = field(default_factory=dict)

    # ---- 自主消息 ----
    _pending_autonomous: List[str] = field(default_factory=list)
    _last_auto_check: float = 0.0
    _last_auto_save: float = 0.0
    _last_save_success: float = 0.0
    _last_triggers: List = field(default_factory=list)
    _chatbot_history: List[Dict] = field(default_factory=list)

    # ---- 训练临时数据 ----
    _extracted_profile: Optional[CharacterProfile] = None
    _last_train_features: Dict = field(default_factory=dict)

    # ---- 可选回调 (供 UI 层注册) ----
    on_state_changed: Optional[Callable] = None   # 情绪状态变化时调用
    on_status_message: Optional[Callable] = None   # 状态消息回调
    on_chatbot_changed: Optional[Callable] = None   # 聊天记录变化时调用

    # ---- 图片生成 ----
    _image_gen_adapter: Any = field(default=None, repr=False)     # 持久存储，不随 agent 重建丢失
    _image_gen_provider: str = field(default="", repr=False)

    def create_agent(self, preset: str, llm_call=None, state_file=None):
        """创建 PersonalityAgent 实例"""
        self.preset_name = preset
        self.agent = create_from_preset(
            preset_name=preset,
            llm_call=llm_call,
            state_file=state_file,
            presets_dir=self.presets_dir,
        )
        self.llm_connected = llm_call is not None
        self._restore_image_gen()

    async def ensure_loaded(self):
        """确保引擎已加载"""
        if self.agent:
            await self.agent.engine.load()

    async def refresh_display_state(self):
        """刷新缓存的状态"""
        if not self.agent:
            return
        self.last_state = await self.agent.get_state()
        self.last_diag = self.agent.get_diagnostics()
        if self.on_state_changed:
            self.on_state_changed(self.last_state, self.last_diag)

    # ---- LLM 连接 ----

    async def connect_llm(self, provider: str, api_key: str, model: str = "", base_url: str = ""):
        """创建 LLM 适配器并重新创建 agent（保留对话历史，含 API Key 验证）"""
        old_history = None
        old_engine_snapshot = None

        if self.agent:
            old_history = list(self.agent.conversation_history)
            old_engine_snapshot = _snapshot_engine_state(self.agent)

        llm_call = None
        if provider == "Anthropic":
            llm_call = await create_anthropic_adapter(api_key=api_key, model=model or "claude-sonnet-4-6", base_url=base_url)
        elif provider == "OpenAI":
            llm_call = await create_openai_adapter(api_key=api_key, model=model or "gpt-4o", base_url=base_url)
        elif provider == "DeepSeek":
            llm_call = await create_deepseek_adapter(api_key=api_key, model=model or "deepseek-chat", base_url=base_url)

        if llm_call is None:
            raise RuntimeError(
                f"无法创建 {provider} 适配器。\n"
                f"请检查是否安装了对应的 SDK 包。"
            )

        # ---- 验证 API Key（发一次真实请求）----
        try:
            await _validate_llm(provider, api_key, model, base_url)
        except Exception as e:
            raise RuntimeError(
                f"API Key 验证失败 — {provider}\n"
                f"Model: {model or '(default)'}\n"
                f"Error: {e}"
            ) from e

        self.create_agent(preset=self.preset_name, llm_call=llm_call)
        await self.ensure_loaded()

        # 恢复对话 + 情感
        if old_history:
            self.agent.conversation_history = old_history
        else:
            restored = _restore_chat_history(self.checkpoint_manager, self.preset_name)
            if restored:
                self.agent.conversation_history = restored

        # 优先用快照（同会话重建），兜底读盘
        if old_engine_snapshot and any(v != 0.0 for v in old_engine_snapshot.values()
                                       if isinstance(v, (int, float))):
            _apply_engine_snapshot(self.agent, old_engine_snapshot)
        else:
            _restore_emotion_state(self.checkpoint_manager, self.preset_name, self.agent)

        await self.refresh_display_state()
        self.llm_provider = provider
        return True

    def disconnect_llm(self):
        """断开 LLM（保留 agent 和对话历史）"""
        if self.agent:
            self.agent._llm_call = None
        self.llm_connected = False
        self.llm_provider = "无"

    # ---- 图片生成 ----

    def connect_image_gen(self, provider: str, api_key: str = "", model: str = "",
                          base_url: str = ""):
        """连接图片生成引擎"""
        from alice.image import create_image_gen
        adapter = create_image_gen(
            provider=provider, api_key=api_key, model=model,
            base_url=base_url, save_dir="images",
        )
        if adapter:
            self._image_gen_adapter = adapter
            self._image_gen_provider = provider
            if self.agent:
                self.agent.set_image_gen(adapter)

    def disconnect_image_gen(self):
        """断开图片生成引擎"""
        self._image_gen_adapter = None
        self._image_gen_provider = ""
        if self.agent:
            self.agent.set_image_gen(None)

    def _restore_image_gen(self):
        """agent 重建后恢复图片引擎"""
        if self._image_gen_adapter and self.agent:
            self.agent.set_image_gen(self._image_gen_adapter)

    # ---- 预设切换 ----

    async def switch_preset(self, target: str):
        """切换人格预设（save-old + load-new）"""
        old_llm = None

        if self.agent:
            old_llm = self.agent._llm_call
            await _auto_save_state_session(self)

        self._pending_autonomous.clear()
        self.create_agent(preset=target, llm_call=old_llm)
        await self.ensure_loaded()

        restored = _restore_chat_history(self.checkpoint_manager, target)
        if restored:
            self.agent.conversation_history = restored
        _restore_emotion_state(self.checkpoint_manager, target, self.agent)

        await self.refresh_display_state()
        return restored

    # ---- 自动保存 ----

    async def auto_save(self):
        await _auto_save_state_session(self)

    # ---- 定时 tick ----

    async def tick(self, minutes: float = 1.0 / 60.0):
        """驱动引擎时间流逝"""
        if not self.agent:
            return
        triggers = await self.agent.tick_time(minutes)
        self._last_triggers = triggers or []
        await self.refresh_display_state()

    async def check_autonomous(self):
        """检查是否触发自主消息（含引擎触发强制检查）"""
        if not self.agent or not self.agent._llm_call:
            return None
        now = time.time()
        if now - self._last_auto_check > 8:
            self._last_auto_check = now
            # 检查引擎是否有强制触发（如 find_activity）
            has_trigger = any(t.get('action') in ('find_activity',)
                              for t in self._last_triggers)
            msg = await self.agent.maybe_initiate(force=has_trigger)
            if msg:
                self._pending_autonomous.append(msg)
                return msg
        return None

    def pop_pending_autonomous(self) -> List[str]:
        """取出并清空待推送的自主消息"""
        msgs = list(self._pending_autonomous)
        self._pending_autonomous.clear()
        return msgs

    # ---- 训练 ----

    def get_or_create_trainer(self):
        """延迟初始化训练器（需要 LLM）"""
        if self.trainer is None:
            llm = self.agent._llm_call if self.agent else None
            self.trainer = PersonalityTrainer(
                llm_call=llm,
                style_store=self.style_store,
                checkpoint_manager=self.checkpoint_manager,
            )
        return self.trainer

    def get_reference_manager(self):
        """延迟初始化参考图素材管理器"""
        if self.reference_manager is None:
            from alice.ui.reference import ReferenceManager
            self.reference_manager = ReferenceManager()
        return self.reference_manager


# ---- 辅助函数 ----


# Session state helpers — extracted to alice_app.session_state
from alice.ui.state import (  # noqa: F401
    validate_llm as _validate_llm,
    snapshot_engine_state as _snapshot_engine_state,
    apply_engine_snapshot as _apply_engine_snapshot,
    restore_chat_history as _restore_chat_history,
    restore_emotion_state as _restore_emotion_state,
    auto_save_state_session as _auto_save_state_session,
)
