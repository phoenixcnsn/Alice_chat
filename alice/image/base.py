"""
图片生成适配器 — 统一抽象接口
"""
import os
import hashlib
import time
from pathlib import Path
from typing import Optional


class ImageGenAdapter:
    """所有图片生成引擎的抽象基类"""

    def __init__(self, save_dir: str = "images"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    async def generate(self, prompt: str, subdir: str = "") -> Optional[str]:
        """
        根据 prompt 生成图片，返回本地文件路径。
        子类必须实现此方法。
        """
        raise NotImplementedError

    async def validate(self) -> None:
        """
        验证连接/API Key 是否有效。
        成功返回 None，失败抛异常（异常消息会显示给用户）。
        子类可重写此方法。
        """
        pass  # 默认不验证

    async def _download_image(self, url: str, subdir: str = "",
                               timeout: float = 120.0) -> str:
        """下载图片到本地，返回文件路径（子类共享方法）"""
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as http:
            resp = await http.get(url)
            resp.raise_for_status()
            path = self._make_path(subdir)
            path.write_bytes(resp.content)
            return str(path)

    def _make_path(self, subdir: str) -> Path:
        """生成唯一的输出文件路径"""
        d = self.save_dir / subdir if subdir else self.save_dir
        d.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        h = hashlib.md5(f"{ts}{os.urandom(4)}".encode()).hexdigest()[:6]
        return d / f"gen_{ts}_{h}.jpg"
