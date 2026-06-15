"""
人格情绪引擎 Agent —— 基于 JiwenEngine 的对话代理

将积温引擎的五轴情绪模型接入 LLM，使 agent 的回复带有持续的、
基于微分方程演化的人格情绪色彩。

轴:
  connection (思念)  — 对数增长，收到回复时缓解
  pride      (骄傲)  — 微分方程，思念驱动 + 回归静息值
  valence    (愉悦度) — 二阶弹簧-阻尼系统
  arousal    (唤醒度) — 资源模型，带冲突激发与沉浸消耗
  immersion  (沉浸度) — 活动维持，自然衰减

用法:
    from alice.core.agent import PersonalityAgent

    agent = PersonalityAgent(
        persona={
            "subjectName": "阿莉丝",
            "selfName": "你",
            "subjectPronoun": "她",
        },
        llm_call=my_async_llm_function,
        base_system_prompt="你是一位名叫 Clara 的 AI 助手...",
    )

    response = await agent.chat("你今天过得怎么样？")
    print(response.text)
    print(response.emotional_state)
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import (
    Any, Awaitable, Callable, Dict, List, Optional, Tuple,
)

from alice.core.engine import JiwenEngine, create_jiwen, clamp


# ------------------------------------------------------------
# 类型定义
# ------------------------------------------------------------
@dataclass
class AgentResponse:
    """Agent 对一条用户消息的完整响应"""
    text: str                                    # LLM 生成的回复文本
    emotional_state: Dict[str, Any]              # 当前情绪状态
    triggers: List[Dict[str, Any]]               # 触发的行为
    style_guidance: str                          # 注入的风格指引
    prompt_context: str                          # 注入的情绪上下文
    raw_llm_response: Any = None                 # LLM 原始响应（调试用）
    image_paths: List[str] = field(default_factory=list)  # ReAct 工具生成的图片路径


@dataclass
class ConversationMessage:
    """对话消息"""
    role: str                                    # "user" | "assistant" | "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    message_id: Optional[str] = None


# ------------------------------------------------------------
# PersonalityAgent
# ------------------------------------------------------------

# Re-exports for backward compatibility — canonical definitions in dedicated modules
from alice.llm.adapters import LLMCall, create_anthropic_adapter, create_openai_adapter, create_deepseek_adapter  # noqa: F401, E501
from alice.llm.prompts import DEFAULT_BASE_SYSTEM_PROMPT, MULTI_BUBBLE_INSTRUCTION, AGENT_TOOLS_INSTRUCTION  # noqa: F401, E501
from alice.storage.file import FilePersistence  # noqa: F401
class PersonalityAgent:
    """基于积温引擎的人格对话代理"""

    def __init__(
        self,
        # ---- 积温引擎配置 ----
        persona: Optional[Dict[str, str]] = None,
        rates: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        engine_opts: Optional[Dict[str, Any]] = None,

        # ---- LLM 配置 ----
        llm_call: Optional[LLMCall] = None,
        base_system_prompt: Optional[str] = None,

        # ---- 持久化 ----
        on_save: Optional[Callable[[Dict], Awaitable[None]]] = None,
        on_load: Optional[Callable[[], Awaitable[Optional[Dict]]]] = None,

        # ---- 记忆系统 ----
        memory_manager: Any = None,         # MemoryManager 实例

        # ---- 人格 RAG 系统 ----
        personality_rag: Any = None,        # PersonalityRAG 实例（风格示例检索）
        style_store: Any = None,            # StyleStore 实例（风格向量存储）

        # ---- 其他 ----
        debug: bool = False,
        max_history_messages: int = 20,
        _character_profile: Any = None,
    ):
        """
        初始化人格 Agent。

        Args:
            persona: 人称配置 {"subjectName": "对方", "selfName": "你", "subjectPronoun": "ta"}
            rates: 覆盖引擎速率参数
            thresholds: 覆盖阈值参数
            engine_opts: 传给 create_jiwen 的完整 options（优先级最高）
            llm_call: 异步 LLM 调用函数 async (system_prompt, messages) -> str
            base_system_prompt: 基础系统提示词模板
            on_save: 状态持久化回调
            on_load: 状态加载回调
            debug: 是否打印调试信息
            max_history_messages: 保留的最大对话历史条数
            _character_profile: (内部) CharacterProfile 实例，用于 few-shot 注入
        """
        self.persona = persona or {
            'subjectName': '对方',
            'selfName': '你',
            'subjectPronoun': 'ta',
        }
        self.debug = debug
        self.max_history_messages = max_history_messages

        # 角色预设
        self._character_profile = _character_profile

        # 记忆系统
        self.memory_manager = memory_manager

        # 人格 RAG 系统
        self.personality_rag = personality_rag
        self.style_store = style_store

        # ---- 构建引擎 ----
        engine_opts = dict(engine_opts or {})
        engine_opts.setdefault('persona', self.persona)
        if rates:
            engine_opts.setdefault('rates', rates)
        if thresholds:
            engine_opts.setdefault('thresholds', thresholds)

        # 持久化回调
        self._user_on_save = on_save
        self._user_on_load = on_load
        engine_opts.setdefault('onSave', self._save_callback)
        engine_opts.setdefault('onLoad', self._load_callback)

        self.engine = create_jiwen(engine_opts)

        # ---- LLM ----
        self._llm_call = llm_call
        self._image_gen = None  # ImageGenAdapter or None
        self.base_system_prompt = base_system_prompt or DEFAULT_BASE_SYSTEM_PROMPT

        # ---- 对话状态 ----
        self.conversation_history: List[ConversationMessage] = []
        self._last_interaction_time: Optional[float] = None

        # ---- 消息桥接：让引擎能感知新消息 ----
        self._pending_incoming_message: Optional[Dict] = None
        self._processed_message_id: Optional[str] = None
        self.engine.get_last_message = self._get_last_message_for_engine

        # ---- saga bias（可由外部故事线设置） ----
        self._saga_bias: Dict[str, float] = {'pride': 0.0, 'valence': 0.0, 'arousal': 0.0}
        self.engine.get_saga_bias = self._get_saga_bias_for_engine

    # --------------------------------------------------------
    # 内部回调
    # --------------------------------------------------------
    async def _save_callback(self, state: Dict):
        """引擎状态变更时触发保存"""
        if self._user_on_save:
            await self._user_on_save(state)

    async def _load_callback(self) -> Optional[Dict]:
        """引擎加载时触发"""
        if self._user_on_load:
            return await self._user_on_load()
        return None

    def _get_last_message_for_engine(self) -> Optional[Dict]:
        """引擎通过此回调检查是否有新消息"""
        return self._pending_incoming_message

    async def _get_saga_bias_for_engine(self) -> Dict:
        """引擎通过此回调获取故事线偏移"""
        return dict(self._saga_bias)

    # --------------------------------------------------------
    # 对话历史管理
    # --------------------------------------------------------
    def _add_to_history(self, role: str, content: str):
        msg = ConversationMessage(role=role, content=content)
        self.conversation_history.append(msg)
        # 裁剪历史
        if len(self.conversation_history) > self.max_history_messages:
            self.conversation_history = self.conversation_history[-self.max_history_messages:]

    def _get_messages_for_llm(self) -> List[Dict[str, str]]:
        """将对话历史转换为 LLM API 格式"""
        return [
            {"role": m.role, "content": m.content}
            for m in self.conversation_history
        ]

    # --------------------------------------------------------
    # 系统提示词构建
    # --------------------------------------------------------
    def _build_system_prompt(self, user_message: str = "") -> str:
        """组装多层级系统提示词：

        Layer 1: 固定人格 (角色预设 / 基础模板)
        Layer 2: 对话记忆 RAG (MemoryManager)
        Layer 3: 情绪状态 (积温引擎)
        Layer 4: 风格示例 RAG (PersonalityRAG — 新增)

        如果配置了 personality_rag，则使用其完整的多层级组装。
        否则回退到原有的简单拼接。
        """
        # 固定人格：角色预设
        if self._character_profile:
            base = self._character_profile.build_system_prompt()
        else:
            base = self.base_system_prompt.format(**self.persona)

        # 多气泡指令（始终追加）
        base += MULTI_BUBBLE_INSTRUCTION

        # 图片生成工具（有图片引擎时注入工具块指令）
        if self._image_gen:
            base += AGENT_TOOLS_INSTRUCTION

        # 动态状态：积温引擎情绪上下文
        prompt_context = self.engine.get_prompt_context()
        style_guidance = self.engine.get_style_guidance()

        # 动态记忆：RAG 检索
        memory_context = ""
        if self.memory_manager and user_message:
            memory_context = self.memory_manager.retrieve_context(user_message)

        # ---- 如果配置了人格 RAG，使用多层级组装 ----
        if self.personality_rag and self._character_profile:
            preset_name = self._character_profile.name
            # 构建包含最近几条历史消息的上下文
            recent_context = user_message
            if self.conversation_history:
                recent = [m.content for m in self.conversation_history[-4:]]
                recent.append(user_message)
                recent_context = "\n".join(recent)

            emotion_context = ""
            if prompt_context:
                emotion_context = f"## 当前感受\n{prompt_context}"
            if style_guidance:
                emotion_context += f"\n\n## 说话风格指引（务必内化，不要逐条复述）\n{style_guidance}"

            return self.personality_rag.build_enriched_system_prompt(
                base_prompt=base,
                preset_name=preset_name,
                current_context=recent_context,
                emotion_context=emotion_context,
                memory_context=memory_context,
                style_example_count=5,
            )

        # ---- 回退到原有简单拼接 ----
        parts = [base]

        if memory_context:
            parts.append(memory_context)

        if prompt_context:
            parts.append(f"\n## 当前感受\n{prompt_context}")

        if style_guidance:
            parts.append(f"\n## 说话风格指引（务必内化，不要逐条复述）\n{style_guidance}")

        return "\n".join(parts)

    # --------------------------------------------------------
    # 核心接口
    # --------------------------------------------------------
    async def chat(self, user_message: str) -> AgentResponse:
        """
        处理一条用户消息并返回带人格的回复。

        这是最主要的交互入口。每次调用会：
        1. 计算距上次交互的时间，驱动情绪演化
        2. 将用户消息注册为"新消息"以触发思念缓解
        3. 构建带情绪上下文的系统提示词
        4. 调用 LLM 生成回复
        5. 更新对话历史和状态
        """
        # ---- 1. 计算时间流逝 ----
        now = time.time()
        if self._last_interaction_time is not None:
            elapsed_seconds = now - self._last_interaction_time
            elapsed_minutes = elapsed_seconds / 60.0
        else:
            elapsed_minutes = 1.0  # 首次交互，假设 1 分钟

        if self.debug:
            print(f"[Agent] 距上次交互: {elapsed_minutes:.1f} 分钟")

        # ---- 2. 将用户消息注册为引擎可见的"新消息" ----
        msg_id = f"msg_{int(now * 1000)}"
        self._pending_incoming_message = {
            'id': msg_id,
            'content': user_message,
            'timestamp': now,
        }

        # ---- 3. 驱动引擎 tick ----
        triggers = await self.engine.tick(elapsed_minutes)

        # ---- 4. 构建系统提示词（传入用户消息以检索记忆） ----
        system_prompt = self._build_system_prompt(user_message)

        style_guidance = self.engine.get_style_guidance()
        prompt_context = self.engine.get_prompt_context()

        if self.debug:
            print(f"[Agent] 系统提示词长度: {len(system_prompt)} 字符")
            print(f"[Agent] 触发: {[t.get('action') for t in triggers]}")

        # ---- 5. 添加到对话历史 ----
        self._add_to_history("user", user_message)

        # ---- 6. 调用 LLM（ReAct 模式） ----
        messages = self._get_messages_for_llm()

        # 首次对话时注入 few-shot 示例
        if self._character_profile and len(self.conversation_history) <= 2:
            examples = self._character_profile.get_examples_as_messages()
            if examples:
                messages = examples + messages

        generated_images: List[str] = []
        _MAX_LOOP = 5  # Agent Loop 最大迭代次数

        if self._llm_call is not None:
            # ================================================================
            # Agent Loop: LLM 纯文本响应 → 解析 ```tool:xxx``` 块 → 执行 → 反馈 → 循环
            # ================================================================
            import re
            loop_msgs = list(messages)

            for _loop in range(_MAX_LOOP):
                llm_response = await self._llm_call(system_prompt, loop_msgs)

                # 解析工具块: ```tool:名称\n{JSON}\n```
                tool_re = re.compile(r'```tool:(\w+)\s*\n(.*?)```', re.DOTALL)
                tool_matches = list(tool_re.finditer(llm_response))

                if not tool_matches:
                    break  # 无工具 → 最终回复

                # 执行工具
                tool_results = []
                for m in tool_matches:
                    t_name = m.group(1)
                    try:
                        t_args = json.loads(m.group(2).strip())
                    except json.JSONDecodeError:
                        t_args = {}
                    result = await self._execute_tool(
                        t_name, t_args, generated_images,
                    )
                    tool_results.append((t_name, result))

                # 去除工具块，保留文本
                cleaned = llm_response
                for m in tool_matches:
                    cleaned = cleaned.replace(m.group(0), "")
                cleaned = cleaned.strip()

                # 反馈给 LLM
                if cleaned:
                    loop_msgs.append({"role": "assistant", "content": cleaned})
                fb = "\n".join(f"[{n} 结果]: {r}" for n, r in tool_results)
                loop_msgs.append({"role": "user", "content": fb})
                print(f"[Agent] Loop {_loop+1}: 执行 {[n for n,_ in tool_results]}")

        else:
            llm_response = self._generate_fallback_response(user_message)

        # ---- 兼容旧 [IMG:] 标记 ----
        if self._image_gen and "[IMG:" in llm_response:
            import re
            m = re.search(r'\[IMG:\s*(.+?)\]', llm_response)
            if m:
                prompt = m.group(1).strip()
                llm_response = re.sub(r'\s*\[IMG:\s*.+?\]\s*', ' ', llm_response).strip()
                preset = self._character_profile.name if self._character_profile else "default"
                print(f"[Agent] [IMG:] 生成: {prompt[:60]}...")
                try:
                    path = await self.image_gen.generate(prompt, preset)
                    generated_images.append(path)
                except Exception as e:
                    print(f"[Agent] [IMG:] 失败: {e}")

        # ---- 8. 记录回复 ----
        self._add_to_history("assistant", llm_response)

        # ---- 9. 更新状态 ----
        self._last_interaction_time = time.time()
        self._processed_message_id = msg_id
        self._pending_incoming_message = None

        # ---- 10. 获取当前情绪状态 ----
        emotional_state = await self.engine.get_state()

        # ---- 11. 持久化记忆 ----
        if self.memory_manager:
            self.memory_manager.save_user_message(user_message, emotional_state)
            self.memory_manager.save_assistant_message(llm_response, emotional_state)
            # 后台: 摘要 + 提取（不阻塞）
            if self._llm_call:
                asyncio.create_task(self._background_memory_tasks())

        # ---- 12. 保存引擎状态 ----
        await self.engine.save()

        return AgentResponse(
            text=llm_response,
            emotional_state=emotional_state,
            triggers=list(triggers),
            style_guidance=style_guidance,
            prompt_context=prompt_context,
            image_paths=generated_images,
        )

    async def _background_memory_tasks(self):
        """后台执行记忆摘要和提取（不阻塞主对话）"""
        try:
            if self.memory_manager and self._llm_call:
                await self.memory_manager.maybe_summarize(self._llm_call)
                await self.memory_manager.background_extract(self._llm_call)
        except Exception:
            pass

    async def tick_time(self, minutes: float) -> List[Dict]:
        """
        纯时间流逝（无用户消息）。
        用于模拟"挂机"期间的情绪演化。

        Returns:
            触发列表
        """
        # 清除待处理消息
        self._pending_incoming_message = None
        triggers = await self.engine.tick(minutes)
        await self.engine.save()
        return triggers

    async def set_activity(self, activity_type: str, label: str = ""):
        """
        设置当前活动。

        Args:
            activity_type: 'reading' | 'search' | 'browse' | 'observe' 或自定义
            label: 活动描述（如 "三体"）
        """
        await self.engine.set_activity(activity_type, label)
        if self.debug:
            print(f"[Agent] 设置活动: {activity_type} ({label})")

    async def apply_event(self, delta: Dict[str, float]):
        """
        施加外部事件对情绪的直接冲击。

        例如故事事件：
            await agent.apply_event({'valence': -0.3, 'pride': 0.1})
        """
        await self.engine.apply_delta(delta)
        if self.debug:
            print(f"[Agent] 施加事件: {delta}")

    # ---- 图片生成 ----
    def set_image_gen(self, adapter):
        """设置图片生成适配器（None 则禁用）"""
        self._image_gen = adapter

    @property
    def image_gen(self):
        return self._image_gen

    async def _execute_tool(
        self, tool_name: str, args: dict, generated_images: List[str],
    ) -> str:
        """执行工具并返回结果字符串。工具集可扩展。"""
        if tool_name == "send_image":
            if not self._image_gen:
                return "Error: 图片引擎未配置"
            prompt = args.get("prompt", "")
            preset = self._character_profile.name if self._character_profile else "default"
            print(f"[Agent] send_image: {prompt[:60]}...")
            try:
                path = await self.image_gen.generate(prompt, preset)
                generated_images.append(path)
                print(f"[Agent] 图片生成完成: {path}")
                return f"图片已生成: {path}"
            except Exception as e:
                print(f"[Agent] 图片生成失败: {e}")
                return f"图片生成失败: {e}"
        return f"未知工具: {tool_name}"

    async def get_state(self) -> Dict[str, Any]:
        """获取当前完整情绪状态"""
        state = await self.engine.get_state()
        # 附加计算 W (净主动意愿)
        conn = state['connection']
        pride = state['pride']
        W = clamp(conn - self.engine.kappa * pride, 0, 1)
        state['net_willingness'] = round(W, 4)
        state['prompt_context'] = self.engine.get_prompt_context()
        state['style_guidance'] = self.engine.get_style_guidance()
        return state

    async def set_saga_bias(self, pride: float = 0.0, valence: float = 0.0, arousal: float = 0.0):
        """设置故事线偏移（用于叙事驱动的情绪基线偏移）"""
        self._saga_bias = {
            'pride': clamp(pride, -1, 1),
            'valence': clamp(valence, -1, 1),
            'arousal': clamp(arousal, -1, 1),
        }

    async def reset_connection(self):
        """重置思念（例如刚聊完天）"""
        await self.engine.reset_connection()

    # --------------------------------------------------------
    # 自主发起对话（"活人感"核心）
    # --------------------------------------------------------
    async def maybe_initiate(self, elapsed_minutes: float = 0.0, force: bool = False) -> Optional[str]:
        """
        检查是否应该主动发起对话，如果是则生成自主消息。

        基于积温引擎的净主动意愿 W = connection - kappa * pride 决定:
          W >= force_contact    → 高概率主动 (60-80%)
          W >= consider_contact → 中概率主动 (15-30%)
          W <  observation      → 几乎不主动

        force=True 时跳过概率检查（引擎有强制触发时使用）。
        """
        if not self._llm_call:
            return None

        # 计算净主动意愿
        conn = self.engine._compute_connection()
        W = clamp(conn - self.engine.kappa * self.engine._pride, 0, 1)

        thresholds = self.engine.thresholds

        # 概率计算
        import random
        roll = random.random()

        if force:
            prob = 1.0  # 强制触发，100%
        elif W >= thresholds.get('force_contact', 0.50):
            prob = 0.35
        elif W >= thresholds.get('consider_contact', 0.35):
            prob = 0.10
        elif W >= thresholds.get('observation', 0.15):
            prob = 0.02
        else:
            prob = 0.0

        # 如果最近刚聊过天（2分钟内有过交互），降低概率
        now = time.time()
        if self._last_interaction_time and (now - self._last_interaction_time) < 120:
            prob *= 0.3

        if roll >= prob:
            return None

        # 生成自主消息
        try:
            # 构建自主消息 prompt — 区别于回复 prompt
            persona = self.persona
            prompt_context = self.engine.get_prompt_context()
            style_guidance = self.engine.get_style_guidance()

            # 基础人格
            if self._character_profile:
                base = self._character_profile.build_system_prompt()
            else:
                base = self.base_system_prompt.format(**self.persona)
            base += MULTI_BUBBLE_INSTRUCTION

            initiative_prompt = f"""你现在**想主动找{persona.get('subjectName', '对方')}说点什么**。这不是在回复谁——是你自己想说。

可以是:
- 一个突然的念头或想法
- 对{persona.get('subjectName', '对方')}的关心或好奇
- 分享你现在在做的事或刚才的经历
- 一个有趣的话题或回忆
- 或者只是单纯想说句话

根据你的性格和当前情绪来决定说什么。不要刻意，自然流露。
不要写"我主动来找你"这类话——直接说你想说的内容就好。
"""

            system_prompt = base + "\n\n" + initiative_prompt

            if prompt_context:
                system_prompt += f"\n\n## 当前感受\n{prompt_context}"
            if style_guidance:
                system_prompt += f"\n\n## 说话风格指引\n{style_guidance}"

            # 最近对话上下文
            recent_msgs = self._get_messages_for_llm()[-6:]

            llm_response_obj = await self._llm_call(system_prompt, recent_msgs)
            llm_response = llm_response_obj if isinstance(llm_response_obj, str) else llm_response_obj.text

            # 记录到对话历史
            self._add_to_history("assistant", llm_response)
            self._last_interaction_time = time.time()

            if self.debug:
                print(f"[Agent] 自主发起 (W={W:.3f}, zone={self.get_diagnostics()['zone']}): {llm_response[:80]}...")

            return llm_response

        except Exception as e:
            if self.debug:
                print(f"[Agent] 自主发起失败: {e}")
            return None

    # --------------------------------------------------------
    # 回退响应（无 LLM 时）
    # --------------------------------------------------------
    def _generate_fallback_response(self, user_message: str) -> str:
        """无 LLM 时，根据情绪状态生成一个描述性的回退响应"""
        state = self.engine._compute_connection()
        p = self.engine._pride
        v = self.engine._valence
        a = self.engine._arousal

        # 用引擎内部的 prompt_context 作为情绪描述
        mood_desc = self.engine.get_prompt_context()

        return (
            f"[无 LLM 模式 — 情绪状态回显]\n"
            f"你说了: 「{user_message}」\n\n"
            f"当前情绪:\n{mood_desc}\n\n"
            f"数值: connection={state:.2f}, pride={p:.2f}, valence={v:.2f}, arousal={a:.2f}"
        )

    # --------------------------------------------------------
    # 重置 & 诊断
    # --------------------------------------------------------
    async def reset(self, hard: bool = False):
        """
        重置 agent 状态。

        Args:
            hard: 如果 True，同时清空对话历史
        """
        if hard:
            self.conversation_history.clear()
        self._last_interaction_time = None
        self._pending_incoming_message = None
        self._saga_bias = {'pride': 0.0, 'valence': 0.0, 'arousal': 0.0}
        # 重建引擎（清空内部状态）
        old_engine = self.engine
        self.engine = create_jiwen({
            'rates': {k: getattr(old_engine, k) for k in dir(old_engine)
                      if k.startswith(('conn_', 'pride_', 'valence_', 'arousal_', 'immersion_',
                                       'activity_', 'kappa')) and not k.startswith('_')},
            'thresholds': dict(old_engine.thresholds),
            'persona': dict(old_engine.persona),
        })
        self.engine.get_last_message = self._get_last_message_for_engine
        self.engine.get_saga_bias = self._get_saga_bias_for_engine
        await self.engine.load()

    def get_diagnostics(self) -> Dict:
        """返回当前诊断信息"""
        conn = self.engine._compute_connection()
        p = self.engine._pride
        v = self.engine._valence
        a = self.engine._arousal
        i = self.engine._immersion
        W = clamp(conn - self.engine.kappa * p, 0, 1)

        thresholds = self.engine.thresholds

        return {
            'connection': round(conn, 4),
            'pride': round(p, 4),
            'valence': round(v, 4),
            'arousal': round(a, 4),
            'immersion': round(i, 4),
            'net_willingness': round(W, 4),
            'virtual_time_min': round(self.engine._virtual_time, 1),
            'zone': (
                'force_contact' if W >= thresholds['force_contact']
                else 'consider_contact' if W >= thresholds['consider_contact']
                else 'observation' if W >= thresholds['observation']
                else 'idle'
            ),
            'mood_quadrant': (
                'high_energy_positive' if v > 0.3 and a > 0.3
                else 'calm_positive' if v > 0.3 and a < -0.3
                else 'agitated_negative' if v < -0.3 and a > 0.3
                else 'low_energy_negative' if v < -0.3 and a < -0.3
                else 'neutral'
            ),
            'last_activity': self.engine._last_activity,
        }


# Adapter factories + FilePersistence moved to:
#   llm_adapters.py, system_prompts.py, file_persistence.py
# Re-exported at top of this file for backward compatibility.


# ------------------------------------------------------------
# 快捷工厂函数
# ------------------------------------------------------------
def create_personality_agent(
    persona_name: str = "默认",
    llm_call: Optional[LLMCall] = None,
    state_file: Optional[str] = None,
    **kwargs,
) -> PersonalityAgent:
    """
    快捷创建 PersonalityAgent（仅内置"默认"预设，其他角色请用 create_from_preset）。

    Args:
        persona_name: 预置人格名（目前仅 "默认"）
        llm_call: LLM 调用函数
        state_file: 状态文件路径（启用文件持久化）
        **kwargs: 传给 PersonalityAgent 的额外参数

    Returns:
        配置好的 PersonalityAgent 实例
    """
    presets = {"默认": {}}
    preset = presets.get(persona_name, presets["默认"])

    final_kwargs = dict(preset)
    final_kwargs.update(kwargs)
    if 'persona' in preset and 'persona' in kwargs:
        final_kwargs['persona'] = {**preset['persona'], **kwargs['persona']}
    if 'rates' in preset and 'rates' in kwargs:
        final_kwargs['rates'] = {**preset['rates'], **kwargs.get('rates', {})}

    final_kwargs['llm_call'] = llm_call

    if state_file:
        persistence = FilePersistence(state_file)
        final_kwargs['on_save'] = persistence.save
        final_kwargs['on_load'] = persistence.load

    return PersonalityAgent(**final_kwargs)


def create_from_preset(
    preset_name: str,
    llm_call: Optional[LLMCall] = None,
    state_file: Optional[str] = None,
    presets_dir: str = "presets",
    use_style_rag: bool = True,
    embed_fn=None,
) -> PersonalityAgent:
    """
    从角色预设文件创建 PersonalityAgent。

    Args:
        preset_name: 预设名（对应 presets/<name>.json）
        llm_call: LLM 调用函数
        state_file: 状态文件路径（启用文件持久化）
        presets_dir: 预设文件目录
        use_style_rag: 是否启用风格 RAG（需要 style_store 中有训练数据）
        embed_fn: embedding 函数

    Returns:
        配置好的 PersonalityAgent 实例
    """
    from alice.storage.preset import PresetManager
    from alice.memory.manager import MemoryManager

    manager = PresetManager(presets_dir)
    profile = manager.load_or_default(preset_name)

    # 记忆系统
    memory_mgr = MemoryManager()
    memory_mgr.preset_name = preset_name

    # 人格 RAG 系统（新增）
    personality_rag = None
    style_store = None
    if use_style_rag and preset_name != "默认":
        try:
            from alice.training.styles import StyleStore
            from alice.training.rag import PersonalityRAG

            style_store = StyleStore(embed_fn=embed_fn)
            # 只有当有风格示例时才启用 RAG
            if style_store.get_example_count(preset_name) > 0:
                personality_rag = PersonalityRAG(style_store)
        except ImportError:
            pass  # ChromaDB 不可用时回退

    # 文件持久化
    on_save = None
    on_load = None
    if state_file:
        persistence = FilePersistence(state_file)
        on_save = persistence.save
        on_load = persistence.load

    # 构建 agent
    agent = PersonalityAgent(
        persona=profile.persona,
        rates=profile.rates,
        llm_call=llm_call,
        base_system_prompt=profile.build_system_prompt(),
        memory_manager=memory_mgr,
        personality_rag=personality_rag,
        style_store=style_store,
        on_save=on_save,
        on_load=on_load,
        _character_profile=profile,
    )
    return agent


# ------------------------------------------------------------
# 自测
# ------------------------------------------------------------
if __name__ == "__main__":
    async def main():
        print("=" * 60)
        print("人格 Agent 自测（无 LLM 模式）")
        print("=" * 60)

        agent = PersonalityAgent(debug=True)

        # 模拟一系列对话
        conversations = [
            "嗨，今天天气不错。",
            "你在干嘛呢？",
            "我想问你一个问题...",
        ]

        for i, msg in enumerate(conversations):
            print(f"\n--- 第 {i+1} 轮 ---")
            print(f"用户: {msg}")
            response = await agent.chat(msg)
            print(f"Clara: {response.text[:200]}...")
            print(f"情绪: c={response.emotional_state['connection']:.2f} "
                  f"p={response.emotional_state['pride']:.2f} "
                  f"v={response.emotional_state['valence']:.2f} "
                  f"a={response.emotional_state['arousal']:.2f}")
            if response.triggers:
                print(f"触发: {[t['action'] for t in response.triggers]}")

            # 模拟间隔
            if i < len(conversations) - 1:
                print("\n[等待 5 分钟...]")
                # 实际测试中这里本该 sleep，但引擎 tick 直接算时间
                agent._last_interaction_time = time.time() - 300  # 假装 5 分钟前

        print("\n" + "=" * 60)
        print("完整状态:")
        state = await agent.get_state()
        for k, v in state.items():
            if not callable(v):
                print(f"  {k}: {v}")

        print("\n诊断:")
        diag = agent.get_diagnostics()
        for k, v in diag.items():
            print(f"  {k}: {v}")

    asyncio.run(main())
