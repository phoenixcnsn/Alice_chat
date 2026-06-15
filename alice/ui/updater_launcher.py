"""
Updater Launcher — 微型更新器
由主程序在下载更新后调用。等待主进程退出 → 替换文件 → 重新启动。
"""

import sys
import time
import shutil
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print("Usage: updater_launcher.py <new_exe> <app_dir>")
        return

    new_exe = Path(sys.argv[1])
    app_dir = Path(sys.argv[2])

    if not new_exe.exists():
        print(f"Update file not found: {new_exe}")
        return

    # 等待主进程退出（最多 10 秒）
    print("Waiting for main app to exit...")
    time.sleep(2)

    # 备份旧文件
    old_exe = app_dir / "alice_chat.exe"
    backup = app_dir / "alice_chat.exe.bak"
    if old_exe.exists():
        if backup.exists():
            backup.unlink()
        shutil.move(str(old_exe), str(backup))

    # 复制新文件
    try:
        shutil.copy(str(new_exe), str(old_exe))
        print(f"Updated: {old_exe}")
    except Exception as e:
        print(f"Update failed: {e}")
        # 恢复备份
        if backup.exists():
            shutil.move(str(backup), str(old_exe))
        return

    # 删除临时文件
    try:
        new_exe.unlink()
    except Exception:
        pass

    # 重新启动
    import subprocess
    subprocess.Popen([str(old_exe)])
    print("Restarting...")


if __name__ == "__main__":
    main()
