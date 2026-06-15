"""
本地 Stable Diffusion 引擎 — 手动放置模型到本地文件夹后加载。
"""
import asyncio
import gc
import warnings
from pathlib import Path
from alice.image.base import ImageGenAdapter

# 关闭 safetensors 和 safety_checker 的警告刷屏
warnings.filterwarnings("ignore", message=".*safety_checker.*")
warnings.filterwarnings("ignore", message=".*safetensors.*")


class DiffusersAdapter(ImageGenAdapter):
    """本地 SD 模型 — 从指定文件夹加载"""

    def __init__(self, model: str = "", save_dir: str = "images",
                 device: str = "", progress_callback=None):
        super().__init__(save_dir)
        self.model_path = model or "models/sd-v1-5"
        self._pipe = None
        self._device = device
        self._loaded = False
        self._progress = progress_callback

    def _report(self, msg: str, done: float = 0, total: float = 0):
        if self._progress:
            self._progress(msg, done, total)

    def load_model(self):
        """加载模型（同步，在 QThread 中调用）。缺失依赖时自动安装。"""
        if self._pipe is not None:
            return self._pipe

        self._report("检测设备...", 0, 1)

        # 自动安装缺失的依赖
        from alice.utils.install import ensure_package
        ensure_package("diffusers")
        ensure_package("transformers")
        ensure_package("accelerate")
        ensure_package("safetensors")

        # torch 需要 CUDA 版本，CPU 版本跑 SD 极慢且吃满 CPU
        import importlib, subprocess, sys
        try:
            importlib.import_module("torch")
        except ImportError:
            print("[Diffusers] 安装 PyTorch CUDA 版本...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu121",
            ])

        import torch
        from diffusers import StableDiffusionPipeline

        if self._device:
            device = self._device
        elif torch.cuda.is_available():
            device = "cuda"
            print(f"[Diffusers] CUDA 可用: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("[Diffusers] ⚠️ CUDA 不可用，使用 CPU (会很慢)。请确认已安装 CUDA 版 PyTorch")

        model_dir = Path(self.model_path).resolve()
        if not model_dir.exists() or not model_dir.is_dir():
            raise FileNotFoundError(
                f"模型文件夹不存在: {model_dir}\n\n"
                "请手动下载 Stable Diffusion 模型到该文件夹:\n"
                "1. 用浏览器/下载工具访问 huggingface.co/runwayml/stable-diffusion-v1-5\n"
                "2. 下载所有文件到本地文件夹\n"
                "3. 在图片生成设置中点击 📁 浏览 选择该文件夹"
            )

        self._report(f"加载模型 {model_dir.name} 到 {device}...", 10, 100)

        dtype = torch.float16 if device == "cuda" else torch.float32
        self._pipe = StableDiffusionPipeline.from_pretrained(
            str(model_dir), torch_dtype=dtype, safety_checker=None,
        )
        self._pipe.to(device)
        self._pipe.enable_attention_slicing()

        self._report("模型就绪", 100, 100)
        self._loaded = True
        return self._pipe

    async def generate(self, prompt: str, subdir: str = "",
                       reference_images: list = None) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._generate_sync, prompt, subdir, reference_images,
        )

    def _generate_sync(self, prompt: str, subdir: str,
                       reference_images: list = None) -> str:
        if not self._loaded or self._pipe is None:
            self.load_model()

        import torch
        from PIL import Image
        import base64, io

        # CLIP token 限制 77，截断过长的 prompt
        full_prompt = self._pipe.tokenizer.encode(
            f"{prompt}, high quality"
        )[:77]
        full_prompt = self._pipe.tokenizer.decode(full_prompt, skip_special_tokens=True)

        # 有参考图 → img2img 模式
        ref_image = None
        if reference_images:
            ref = reference_images[0]
            if isinstance(ref, str) and ref.startswith("data:"):
                # base64 data URI
                _, b64 = ref.split(",", 1)
                ref_image = Image.open(io.BytesIO(base64.b64decode(b64)))
            elif isinstance(ref, str):
                ref_image = Image.open(ref)

        if ref_image:
            # img2img: 参考图 + 噪声 → 新图
            ref_image = ref_image.convert("RGB").resize((512, 512))
            image = self._pipe(
                prompt=full_prompt,
                image=ref_image,
                strength=0.65,  # 0.65=保留35%原图结构
                negative_prompt="blurry, low quality, ugly, distorted",
                num_inference_steps=25, guidance_scale=7.5,
            ).images[0]
        else:
            image = self._pipe(
                prompt=full_prompt,
                negative_prompt="blurry, low quality, ugly, distorted",
                width=512, height=512, num_inference_steps=20, guidance_scale=7.5,
            ).images[0]

        path = self._make_path(subdir)
        image.save(str(path), format="JPEG", quality=88)

        torch.cuda.empty_cache()
        gc.collect()
        return str(path)
