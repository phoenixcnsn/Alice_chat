"""
会话状态辅助函数 — 快照 / 恢复 / 验证
从 app_session.py 提取
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from alice.ui.session import AgentSession

from alice.core.agent import ConversationMessage


# 引擎状态 key 列表（引擎属性 -> JSON key）
ENGINE_STATE_KEYS = [
    ('_virtual_time', 'virtual_time'),
    ('_pride', 'pride'),
    ('_valence', 'valence'),
    ('_valence_vel', 'valence_vel'),
    ('_arousal', 'arousal'),
    ('_immersion', 'immersion'),
]


async def validate_llm(provider: str, api_key: str, model: str, base_url: str):
    """验证 LLM API Key 是否有效 — 发一次轻量请求"""
    if provider == "Anthropic":
        import anthropic
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = anthropic.AsyncAnthropic(**kwargs)
        # 用 max_tokens=1 做最小验证
        await client.messages.create(
            model=model or "claude-sonnet-4-6",
            max_tokens=1,
            messages=[{"role": "user", "content": "."}],
        )

    elif provider in ("OpenAI", "DeepSeek"):
        from alice.utils.install import ensure_package
        ensure_package("openai")
        from openai import AsyncOpenAI
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        elif provider == "DeepSeek":
            kwargs["base_url"] = "https://api.deepseek.com/v1"
        client = AsyncOpenAI(**kwargs)
        # models.list() 免费，适合验证
        await client.models.list()

    else:
        raise ValueError(f"未知的 LLM provider: {provider}")


def snapshot_engine_state(agent) -> dict:
    """快照引擎情感状态"""
    try:
        eng = agent.engine
        return {
            "virtual_time": getattr(eng, '_virtual_time', 0.0),
            "pride": getattr(eng, '_pride', 0.0),
            "valence": getattr(eng, '_valence', 0.0),
            "valence_vel": getattr(eng, '_valence_vel', 0.0),
            "arousal": getattr(eng, '_arousal', 0.0),
            "immersion": getattr(eng, '_immersion', 0.0),
            "last_activity": getattr(eng, '_last_activity', None),
        }
    except Exception as e:
        print(f"[App] 快照引擎状态失败: {e}")
        return {}


def apply_engine_snapshot(agent, snapshot: dict):
    """应用快照到引擎"""
    if not snapshot or not agent or not agent.engine:
        return
    try:
        engine = agent.engine
        for attr, _key in ENGINE_STATE_KEYS:
            if attr in snapshot and hasattr(engine, attr):
                setattr(engine, attr, snapshot[attr])
        if '_last_activity' in snapshot and hasattr(engine, '_last_activity'):
            engine._last_activity = snapshot['_last_activity']
        engine._loaded = True
    except Exception as e:
        print(f"[App] 应用快照失败: {e}")


def restore_chat_history(checkpoint_mgr, preset_name: str) -> List[ConversationMessage]:
    """从存档恢复对话历史"""
    try:
        es = checkpoint_mgr.load_engine_state(preset_name)
        if es and es.get("chat_history"):
            restored = []
            for m in es["chat_history"]:
                restored.append(ConversationMessage(
                    role=m.get("role", ""),
                    content=m.get("content", ""),
                    timestamp=m.get("timestamp", time.time()),
                ))
            print(f"[App] 已恢复 {len(restored)} 条对话 for {preset_name}")
            return restored
    except Exception as e:
        print(f"[App] 恢复对话失败: {e}")
    return []


def restore_emotion_state(checkpoint_mgr, preset_name: str, agent):
    """从存档恢复引擎情感状态"""
    if not agent or not agent.engine:
        return
    try:
        es = checkpoint_mgr.load_engine_state(preset_name)
        if not es:
            return
        engine = agent.engine
        for attr, key in ENGINE_STATE_KEYS:
            if hasattr(engine, attr):
                setattr(engine, attr, es.get(key, 0.0))
        last_act = es.get("last_activity")
        if last_act and last_act != "None" and hasattr(engine, '_last_activity'):
            engine._last_activity = last_act
        engine._loaded = True
        print(f"[App] 磁盘恢复状态 for {preset_name}: pride={engine._pride:.2f}, "
              f"valence={engine._valence:.2f}, vt={engine._virtual_time:.1f}min")
    except Exception as e:
        print(f"[App] 磁盘恢复状态失败 for {preset_name}: {e}")


async def auto_save_state_session(session: AgentSession):
    """保存当前 agent 的完整状态"""
    agent = session.agent
    if not agent or not agent._character_profile:
        return
    preset_name = agent._character_profile.name
    try:
        await session.checkpoint_manager.save_full_checkpoint(
            profile=agent._character_profile,
            agent=agent,
        )
        session._last_save_success = time.time()
    except Exception:
        pass
