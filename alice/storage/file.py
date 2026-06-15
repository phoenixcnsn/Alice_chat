"""
简单 JSON 文件持久化 — 从 personality_agent.py 提取
"""
import json
import os
from typing import Dict, Optional


class FilePersistence:
    """基于 JSON 文件的状态持久化"""

    def __init__(self, filepath: str):
        self.filepath = filepath

    async def save(self, state: Dict):
        """保存状态到文件"""
        os.makedirs(os.path.dirname(self.filepath) or '.', exist_ok=True)
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    async def load(self) -> Optional[Dict]:
        """从文件加载状态"""
        if not os.path.exists(self.filepath):
            return None
        with open(self.filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
