"""
图片生成器统一入口

用法:
    from alice.image import create_image_gen

    gen = create_image_gen("replicate", api_key="r8_xxx")
    path = await gen.generate("a beautiful sunset over the ocean")
"""

from typing import Optional
from alice.image.base import ImageGenAdapter


def create_image_gen(
    provider: str,
    api_key: str = "",
    model: str = "",
    base_url: str = "",
    save_dir: str = "images",
) -> Optional[ImageGenAdapter]:
    """
    根据 provider 创建对应的图片生成适配器。

    provider 可选值:
        "replicate"   — Replicate (SDXL)
        "openai"      — OpenAI DALL-E 3
        "sdwebui"     — 本地 AUTOMATIC1111 / ComfyUI
        "diffusers"   — 本地 diffusers 库 (需要 torch)
        "none" / ""   — 不启用
    """
    provider = provider.lower().strip()

    if provider in ("none", ""):
        return None

    if provider == "replicate":
        from alice.image.replicate import ReplicateAdapter
        return ReplicateAdapter(
            api_key=api_key,
            model=model or "black-forest-labs/flux-2-pro",
            save_dir=save_dir,
        )

    if provider == "openai":
        from alice.image.openai import OpenAIImageAdapter
        return OpenAIImageAdapter(
            api_key=api_key,
            model=model or "dall-e-3",
            save_dir=save_dir,
            base_url=base_url,
        )

    if provider == "sdwebui":
        from alice.image.sdwebui import SDWebUIAdapter
        return SDWebUIAdapter(
            base_url=base_url or "http://localhost:7860",
            save_dir=save_dir,
        )

    if provider == "diffusers":
        try:
            from alice.image.diffusers import DiffusersAdapter
            return DiffusersAdapter(
                model=model or "runwayml/stable-diffusion-v1-5",
                save_dir=save_dir,
            )
        except ImportError as e:
            raise ImportError(
                "本地 diffusers 需要安装额外依赖:\n"
                "  pip install torch diffusers transformers accelerate safetensors"
            ) from e

    raise ValueError(f"未知的图片生成引擎: {provider}")
