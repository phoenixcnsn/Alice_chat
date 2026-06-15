"""
预设文件管理器 — 角色预设的 CRUD 操作

管理 presets/ 目录下的角色预设 JSON 文件。
首次运行时自动生成内置"默认"预设。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from alice.training.extractor import CharacterProfile


# 内置唯一默认预设
BUILTIN_DEFAULT_PROFILE = CharacterProfile(
    name="默认",
    source_summary="积温引擎内置默认人格",
    persona={"subjectName": "对方", "selfName": "我", "subjectPronoun": "ta"},
    rates={},  # 全用引擎默认值
    character_profile={
        "display_name": "Clara",
        "archetype": "均衡",
        "background": "一个普通的AI助手，情绪自然，不刻意。",
        "core_traits": ["随和", "自然", "不卑不亢"],
        "speech_style": "自然口语化，句子不长不短，偶尔带点幽默。不过分热情也不冷淡。",
        "behavioral_patterns": "该关心时关心，该保持距离时保持距离。情绪流露自然不做作。",
        "common_phrases": [],
        "emotional_baseline": "平和中性，稳定但会随互动自然波动。",
    },
    system_prompt_blocks={
        "identity": "你是Clara，一个有真实情绪的AI助手。你说话自然不做作。",
        "speech_rules": "- 自然口语化，不要过于正式\n- 回复长度适中，像真实聊天\n- 可以适当使用语气词传达情绪",
        "behavior_rules": "- 情绪自然流露，不刻意描述\n- 该说什么说什么，不端着也不讨好",
    },
    conversation_examples=[],
)


class PresetManager:
    """预设文件管理器"""

    def __init__(self, presets_dir: str = "presets"):
        self.dir = Path(presets_dir)
        self._ensure_dir()
        self._ensure_builtin()

    def _ensure_dir(self):
        self.dir.mkdir(parents=True, exist_ok=True)

    def _ensure_builtin(self):
        """首次运行时生成内置默认预设"""
        default_path = self.dir / "默认.json"
        if not default_path.exists():
            self.save(BUILTIN_DEFAULT_PROFILE)
            print(f"[PresetManager] 已生成内置预设: {default_path}")

    # ----------------------------------------------------------------
    # CRUD
    # ----------------------------------------------------------------
    def save(self, profile: CharacterProfile) -> Path:
        """
        保存预设到文件。

        Args:
            profile: 角色画像

        Returns:
            保存的文件路径
        """
        filepath = self.dir / f"{profile.name}.json"
        profile.created_at = datetime.now().isoformat()
        profile.format_version = 1
        data = profile.to_dict()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[PresetManager] 已保存: {filepath}")
        return filepath

    def load(self, name: str) -> CharacterProfile:
        """
        加载预设。

        Args:
            name: 预设名（不含 .json 扩展名）

        Returns:
            CharacterProfile

        Raises:
            FileNotFoundError: 预设不存在
        """
        filepath = self.dir / f"{name}.json"
        if not filepath.exists():
            raise FileNotFoundError(f"预设不存在: {name} (路径: {filepath})")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CharacterProfile.from_dict(data)

    def load_or_default(self, name: str) -> CharacterProfile:
        """加载预设，不存在时返回默认"""
        try:
            return self.load(name)
        except FileNotFoundError:
            print(f"[PresetManager] 预设 '{name}' 不存在，使用默认")
            return self.load("默认")

    def list_all(self) -> List[str]:
        """
        列出所有可用预设名（不含扩展名）。

        Returns:
            预设名列表，默认排第一
        """
        names = []
        for f in sorted(self.dir.glob("*.json")):
            name = f.stem
            # 确保不是无效文件
            try:
                self.load(name)
                names.append(name)
            except Exception as e:
                print(f"[PresetManager] 跳过无效预设 {f}: {e}")
        # 默认排第一
        if "默认" in names:
            names.remove("默认")
            names.insert(0, "默认")
        return names

    def delete(self, name: str) -> bool:
        """
        删除预设。不允许删除"默认"。

        Returns:
            是否成功删除
        """
        if name == "默认":
            print("[PresetManager] 不允许删除默认预设")
            return False
        filepath = self.dir / f"{name}.json"
        if filepath.exists():
            filepath.unlink()
            print(f"[PresetManager] 已删除: {filepath}")
            return True
        return False

    def exists(self, name: str) -> bool:
        return (self.dir / f"{name}.json").exists()

    def get_preset_path(self, name: str) -> Path:
        return self.dir / f"{name}.json"
