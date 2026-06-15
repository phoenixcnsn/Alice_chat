"""
Updater — 后台自动更新检查与下载
启动时检查 GitHub Releases，提示用户更新。
"""
import json
import hashlib
import tempfile
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urlparse

# 配置
UPDATE_REPO = "owner/alice-chat"       # GitHub 仓库名
UPDATE_VERSION = "0.1.0"              # 当前版本
UPDATE_CHECK_URL = f"https://api.github.com/repos/{UPDATE_REPO}/releases/latest"


def _get_latest_release() -> dict:
    """获取最新 release 信息"""
    try:
        req = Request(UPDATE_CHECK_URL, headers={"Accept": "application/vnd.github+json",
                                                  "User-Agent": "Alice-Chat-Updater"})
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return {}


def _download_file(url: str, dest: Path) -> bool:
    """下载文件到目标路径"""
    try:
        req = Request(url, headers={"User-Agent": "Alice-Chat-Updater"})
        with urlopen(req, timeout=300) as resp:
            dest.write_bytes(resp.read())
        return True
    except Exception:
        return False


def _verify_sha256(filepath: Path, expected: str) -> bool:
    """校验文件哈希"""
    try:
        h = hashlib.sha256(filepath.read_bytes()).hexdigest()
        return h == expected
    except Exception:
        return False


async def check_update_async(main_window):
    """异步检查更新（在后台 asyncio 线程中运行）"""
    release = _get_latest_release()
    if not release:
        return

    tag = release.get("tag_name", "").lstrip("v")
    if not tag:
        return

    # 比较版本（简单字符串比较）
    if tag <= UPDATE_VERSION:
        return

    # 找到 .exe 下载链接
    assets = release.get("assets", [])
    exe_url = None
    exe_sha = None
    for a in assets:
        name = a.get("name", "")
        if name.endswith(".exe"):
            exe_url = a.get("browser_download_url")
            # SHA256 可能在 body 中或者没有
            body = release.get("body", "")
            if f"sha256:{name}" in body:
                for line in body.split("\n"):
                    if f"sha256:{name}" in line:
                        exe_sha = line.split("sha256:")[-1].strip().split()[0]

    if not exe_url:
        return

    # 通知主窗口
    main_window.show_update_notification(tag, exe_url)


async def download_and_apply_update(download_url: str):
    """下载并应用更新"""
    tmp = Path(tempfile.gettempdir()) / "alice_chat_update.exe"
    if tmp.exists():
        tmp.unlink()

    ok = _download_file(download_url, tmp)
    if not ok:
        return

    # 启动 updater_launcher
    launcher = Path(sys.executable).parent / "updater_launcher.exe"
    if not launcher.exists():
        # 开发模式：直接用 Python 启动
        launcher_script = Path(__file__).parent / "updater_launcher.py"
        subprocess.Popen(
            [sys.executable, str(launcher_script), str(tmp),
             str(Path(sys.executable).parent)],
            creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.DETACHED_PROCESS
            if sys.platform == "win32" else 0,
        )
    else:
        subprocess.Popen(
            [str(launcher), str(tmp), str(Path(sys.executable).parent)],
            creationflags=subprocess.DETACHED_PROCESS if sys.platform == "win32" else 0,
        )

    # 退出主程序
    sys.exit(0)
