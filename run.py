"""Alice Chat — 桌面应用启动器

用法:
    python run.py             终端启动（无控制台窗口）
    启动.bat                  双击启动

始终使用 .venv 中的 pythonw.exe，终端和 bat 效果一致。
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
VENV_DIR = ROOT / ".venv" / "Scripts"
PYTHONW = VENV_DIR / "pythonw.exe"
MAIN = ROOT / "main.py"

if not PYTHONW.exists():
    print(f"错误: 未找到 {PYTHONW}")
    print("请先创建虚拟环境: python -m venv .venv")
    print("然后安装依赖: .venv\\Scripts\\pip install -r requirements.txt")
    sys.exit(1)

if not MAIN.exists():
    print(f"错误: 未找到 {MAIN}")
    sys.exit(1)

# 启动桌面应用（无控制台窗口）
creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
subprocess.Popen(
    [str(PYTHONW), str(MAIN)],
    creationflags=creationflags,
    cwd=str(ROOT),
)
