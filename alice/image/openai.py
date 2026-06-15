"""OpenAI DALL-E 3 适配器"""
import httpx
from alice.image.base import ImageGenAdapter


class OpenAIImageAdapter(ImageGenAdapter):
    def __init__(self, api_key: str, model: str = "dall-e-3",
                 size: str = "1024x1024", quality: str = "standard",
                 save_dir: str = "images", base_url: str = ""):
        super().__init__(save_dir)
        self.api_key = api_key
        self.model = model
        self.size = size
        self.quality = quality
        self.base_url = (base_url.rstrip("/") + "/v1") if base_url else "https://api.openai.com/v1"

    async def generate(self, prompt: str, subdir: str = "") -> str:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.base_url}/images/generations",
                json={"model": self.model, "prompt": prompt,
                      "size": self.size, "quality": self.quality, "n": 1},
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
            img_url = data["data"][0]["url"]

            # 下载图片（使用基类共享方法）
            return await self._download_image(img_url, subdir)
