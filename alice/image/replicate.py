"""Replicate API 适配器 — 使用官方 replicate SDK"""
import asyncio
import httpx
from pathlib import Path
from alice.image.base import ImageGenAdapter

# 不同模型的默认输入参数
_MODEL_DEFAULTS = {
    "black-forest-labs/flux-2-pro": {
        "resolution": "1 MP",
        "aspect_ratio": "1:1",
        "output_format": "webp",
        "output_quality": 80,
        "safety_tolerance": 2,
    },
    # SDXL 系列
    "stability-ai/sdxl": {
        "width": 1024,
        "height": 1024,
        "num_outputs": 1,
        "negative_prompt": "blurry, low quality, distorted",
        "scheduler": "K_EULER",
    },
}


class ReplicateAdapter(ImageGenAdapter):
    def __init__(self, api_key: str, model: str = "black-forest-labs/flux-2-pro",
                 save_dir: str = "images"):
        super().__init__(save_dir)
        self.api_key = api_key
        # 自动修正旧模型名
        if model in ("stability-ai/stable-diffusion-xl", ""):
            model = "black-forest-labs/flux-2-pro"
        self.model = model

    async def validate(self) -> None:
        """验证 API Key 是否有效 — 调用需要认证的 account 接口"""
        if not self.api_key or not self.api_key.startswith("r8_"):
            raise ValueError(f"API Key 格式无效 (应以 r8_ 开头): {self.api_key[:8]}...")

        import httpx
        async with httpx.AsyncClient(timeout=15) as http:
            # 用 /account 端点验证（必须认证，不会假通过）
            resp = await http.get(
                "https://api.replicate.com/v1/account",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            if resp.status_code == 401:
                body = _safe_response_text(resp)
                raise RuntimeError(
                    f"API Key 无效 (401 Unauthorized)\n"
                    f"请检查 Key 是否正确: 以 r8_ 开头，共约 40 个字符\n"
                    f"Response: {body}"
                )
            elif not resp.is_success:
                body = _safe_response_text(resp)
                raise RuntimeError(
                    f"Replicate API 连接失败 (HTTP {resp.status_code})\n"
                    f"Response: {body}"
                )
            data = resp.json()
            username = data.get("username", "?")
            print(f"[Replicate] 验证成功: account={username}")

    def _build_input(self, prompt: str, reference_base64: list = None) -> dict:
        """根据模型构建合适的 input 参数，可选合并参考图"""
        defaults = _MODEL_DEFAULTS.get(self.model, {})
        inp = {"prompt": prompt}
        inp.update(defaults)
        if reference_base64:
            inp["input_images"] = reference_base64
        return inp

    async def generate(self, prompt: str, subdir: str = "",
                       reference_images: list = None) -> str:
        """
        生成图片。

        Args:
            prompt: 图片描述
            subdir: 保存子目录
            reference_images: base64 data URI 列表，作为人物/风格参考
        """
        if not self.api_key or not self.api_key.startswith("r8_"):
            raise RuntimeError(
                f"Replicate API Key 未设置或无效 (以 r8_ 开头)\n"
                f"请在 设置面板 → 图片生成 → 输入 API Key → 点击「连接图片引擎」"
            )

        from alice.utils.install import ensure_package
        ensure_package("replicate")
        import replicate

        client = replicate.Client(api_token=self.api_key)
        inp = self._build_input(prompt, reference_images)

        try:
            # SDK 是同步的，放到线程中执行避免阻塞事件循环
            output = await asyncio.to_thread(
                client.run, self.model, input=inp, use_file_output=False
            )
        except replicate.exceptions.ReplicateError as e:
            raise RuntimeError(
                f"Replicate 生成失败\n"
                f"Model: {self.model}\n"
                f"Error: {e}"
            ) from e

        # 提取图片 URL
        img_url = _extract_url(output)
        if not img_url:
            raise RuntimeError(
                f"Replicate 返回了无法处理的输出格式\n"
                f"Model: {self.model}\n"
                f"Output: {output}"
            )

        # 下载图片（使用基类共享方法）
        return await self._download_image(img_url, subdir)


def _safe_response_text(resp) -> str:
    """安全提取响应文本，截断过长内容"""
    try:
        text = resp.text
    except Exception:
        text = "(无法读取响应内容)"
    if len(text) > 500:
        text = text[:500] + "..."
    return text


def _extract_url(output) -> str | None:
    """从 Replicate 返回的各种格式中提取图片 URL"""
    # FileOutput 对象
    if hasattr(output, 'url'):
        return str(output.url)
    # 字符串 URL
    if isinstance(output, str) and output.startswith(("http://", "https://")):
        return output
    # 列表
    if isinstance(output, list) and output:
        return _extract_url(output[0])
    return None
