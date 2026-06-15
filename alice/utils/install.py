"""
自动检测并安装缺失的 Python 包。

当 import 失败时，自动运行 pip install，无需用户手动操作。
"""

import subprocess
import sys
import importlib
from typing import Optional


def ensure_package(import_name: str, pip_name: str = ""):
    """确保 Python 包已安装，如缺失则自动 pip install。

    Args:
        import_name: import 时使用的包名（如 "openai"）
        pip_name: pip install 时的包名，默认同 import_name

    Returns:
        module: 成功导入的模块对象

    Raises:
        ImportError: 自动安装失败时抛出，附带手动安装提示
    """
    pkg = pip_name or import_name

    try:
        return importlib.import_module(import_name)
    except ImportError:
        print(f"\n[auto-install] 检测到缺少依赖: {pkg}")
        print(f"[auto-install] 正在自动安装 {pkg} ...")

        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg],
                stdout=sys.stderr,
                stderr=sys.stderr,
            )
            print(f"[auto-install] ✓ {pkg} 安装成功\n")
        except subprocess.CalledProcessError as e:
            raise ImportError(
                f"自动安装 {pkg} 失败。请手动运行:\n"
                f"  {sys.executable} -m pip install {pkg}"
            ) from e

        # 安装成功后重新导入
        try:
            return importlib.import_module(import_name)
        except ImportError as e:
            raise ImportError(
                f"{pkg} 已安装但仍无法导入，请检查 Python 环境:\n"
                f"  {sys.executable} -m pip install {pkg}"
            ) from e
