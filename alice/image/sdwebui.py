"""本地 SD WebUI API 适配器 (AUTOMATIC1111 / ComfyUI)"""
import base64
import httpx
from alice.image.base import ImageGenAdapter


class SDWebUIAdapter(ImageGenAdapter):
    """连接已运行的 AUTOMATIC1111 或 ComfyUI WebUI"""

    def __init__(self, base_url: str = "http://localhost:7860",
                 steps: int = 20, width: int = 768, height: int = 768,
                 save_dir: str = "images"):
        super().__init__(save_dir)
        self.base_url = base_url.rstrip("/")
        self.steps = steps
        self.width = width
        self.height = height
        self._comfy_mode = False

    async def generate(self, prompt: str, subdir: str = "") -> str:
        async with httpx.AsyncClient(timeout=180) as client:
            # 先试 A1111 API
            try:
                result = await self._generate_a1111(client, prompt)
                if result:
                    return result
            except Exception:
                pass

            # 回退到 ComfyUI
            try:
                result = await self._generate_comfy(client, prompt)
                if result:
                    return result
            except Exception:
                pass

            raise RuntimeError(
                f"无法连接到 SD WebUI: {self.base_url}\n"
                "请确认以下之一已启动：\n"
                "  AUTOMATIC1111: python launch.py --api --listen --port 7860\n"
                "  ComfyUI: python main.py --listen --port 8188"
            )

    async def _generate_a1111(self, client: httpx.AsyncClient, prompt: str) -> str:
        resp = await client.post(
            f"{self.base_url}/sdapi/v1/txt2img",
            json={"prompt": prompt, "negative_prompt": "blurry, low quality, ugly, distorted",
                  "steps": self.steps, "width": self.width, "height": self.height,
                  "restore_faces": True},
        )
        if resp.status_code != 200:
            return ""
        data = resp.json()
        img_bytes = base64.b64decode(data["images"][0])
        path = self._make_path(subdir)
        path.write_bytes(img_bytes)
        return str(path)

    async def _generate_comfy(self, client: httpx.AsyncClient, prompt: str) -> str:
        resp = await client.get(f"{self.base_url}/prompt")
        if resp.status_code != 200:
            return ""

        import json
        workflow = {
            "3": {"inputs": {"seed": 0, "steps": self.steps, "cfg": 7,
                   "sampler_name": "euler", "scheduler": "normal",
                   "denoise": 1.0, "model": ["4", 0],
                   "positive": ["6", 0], "negative": ["7", 0],
                   "latent_image": ["5", 0]}, "class_type": "KSampler"},
            "4": {"inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
                  "class_type": "CheckpointLoaderSimple"},
            "5": {"inputs": {"width": self.width, "height": self.height,
                   "batch_size": 1}, "class_type": "EmptyLatentImage"},
            "6": {"inputs": {"text": prompt, "clip": ["4", 1]},
                  "class_type": "CLIPTextEncode"},
            "7": {"inputs": {"text": "blurry, low quality, ugly",
                   "clip": ["4", 1]}, "class_type": "CLIPTextEncode"},
            "8": {"inputs": {"samples": ["3", 0], "vae": ["4", 2]},
                  "class_type": "VAEDecode"},
            "9": {"inputs": {"filename_prefix": "alice_gen",
                   "images": ["8", 0]}, "class_type": "SaveImage"},
        }
        resp = await client.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow},
        )
        if resp.status_code != 200:
            return ""

        prompt_id = resp.json()["prompt_id"]

        # 轮询等待完成
        import asyncio
        for _ in range(80):
            await asyncio.sleep(1.5)
            hist_resp = await client.get(f"{self.base_url}/history/{prompt_id}")
            if hist_resp.status_code == 200:
                hist = hist_resp.json()
                if prompt_id in hist:
                    outputs = hist[prompt_id].get("outputs", {})
                    for node_id, node_out in outputs.items():
                        images = node_out.get("images", [])
                        if images:
                            img_info = images[0]
                            img_path = img_info.get("filename", img_info.get("name", ""))
                            if img_path:
                                full_path = self.save_dir / img_path
                                if full_path.exists():
                                    return str(full_path)
            else:
                break

        return ""
