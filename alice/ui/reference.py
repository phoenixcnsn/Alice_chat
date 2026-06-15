"""
人物风格参考图素材管理器
- 存储: references/<preset_name>/*.jpg...
- 用途: 图片生成时作为风格/人物参考（类比文本训练的对话示例）
"""
import base64
import shutil
import zipfile
from pathlib import Path
from typing import List

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


class ReferenceManager:
    """人物风格参考图素材库"""

    def __init__(self, base_dir: str = "references"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ---- 查询 ----

    def get_images(self, preset_name: str) -> List[Path]:
        """获取某个角色的所有素材图路径"""
        folder = self.base_dir / preset_name
        if not folder.is_dir():
            return []
        images = []
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(f)
        return images

    def image_count(self, preset_name: str) -> int:
        """素材图数量"""
        return len(self.get_images(preset_name))

    # ---- 添加 ----

    def add_images(self, preset_name: str, file_paths: List[str]) -> int:
        """复制图片文件到素材库，返回成功添加的数量"""
        folder = self.base_dir / preset_name
        folder.mkdir(parents=True, exist_ok=True)
        added = 0
        for fp in file_paths:
            src = Path(fp)
            if not src.is_file() or src.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            dst = folder / src.name
            # 文件名冲突时加序号
            if dst.exists():
                stem = src.stem
                i = 1
                while dst.exists():
                    dst = folder / f"{stem}_{i}{src.suffix}"
                    i += 1
            shutil.copy2(src, dst)
            added += 1
        return added

    def extract_zip(self, preset_name: str, zip_path: str) -> int:
        """
        解压 ZIP 文件，递归扫描所有子文件夹，提取所有图片。
        返回提取的图片数量。
        """
        folder = self.base_dir / preset_name
        folder.mkdir(parents=True, exist_ok=True)
        added = 0
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                name = Path(member.filename)
                if name.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                # 只取文件名，丢弃 zip 内的目录结构
                dst = folder / name.name
                if dst.exists():
                    stem = name.stem
                    i = 1
                    while dst.exists():
                        dst = folder / f"{stem}_{i}{name.suffix}"
                        i += 1
                with zf.open(member) as src, open(dst, 'wb') as out:
                    out.write(src.read())
                added += 1
        return added

    # ---- 删除 ----

    def delete_image(self, preset_name: str, filename: str) -> bool:
        """删除单张素材图"""
        target = self.base_dir / preset_name / filename
        if target.is_file():
            target.unlink()
            return True
        return False

    def delete_all(self, preset_name: str) -> int:
        """删除某角色的全部素材图"""
        folder = self.base_dir / preset_name
        if not folder.is_dir():
            return 0
        count = self.image_count(preset_name)
        shutil.rmtree(folder)
        return count

    # ---- 工具 ----

    @staticmethod
    def to_base64(image_path: Path) -> str:
        """图片文件 → base64 data URI"""
        ext = image_path.suffix.lower()
        mime = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp",
            ".bmp": "image/bmp", ".gif": "image/gif",
        }.get(ext, "image/jpeg")
        data = image_path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"
