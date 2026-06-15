"""
Alice Chat — 积温情绪引擎桌面版入口
"""
import sys
import os
import asyncio
import threading
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
warnings.filterwarnings("ignore", message=".*sm_120.*")

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject

from alice.ui.session import AgentSession
from alice.ui.window import MainWindow


class _CallbackBridge(QObject):
    sig_result = pyqtSignal(object, object)


def setup_theme(app: QApplication):
    app.setStyle("Fusion")
    palette = app.palette()
    palette.setColor(palette.Window, Qt.black)
    palette.setColor(palette.WindowText, Qt.white)
    palette.setColor(palette.Base, Qt.black)
    palette.setColor(palette.AlternateBase, Qt.darkGray)
    palette.setColor(palette.ToolTipBase, Qt.black)
    palette.setColor(palette.ToolTipText, Qt.white)
    palette.setColor(palette.Text, Qt.white)
    palette.setColor(palette.Button, Qt.darkGray)
    palette.setColor(palette.ButtonText, Qt.white)
    palette.setColor(palette.Highlight, Qt.darkCyan)
    palette.setColor(palette.HighlightedText, Qt.black)
    app.setPalette(palette)

    qss_path = Path(__file__).parent / "alice" / "ui" / "resources" / "style.qss"
    if qss_path.exists():
        try:
            app.setStyleSheet(qss_path.read_text(encoding="utf-8"))
        except Exception:
            pass


class AsyncLoopThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.loop: asyncio.AbstractEventLoop = None
        self._bridge: _CallbackBridge = None

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run_coro(self, coro, callback=None):
        bridge = self._bridge

        async def _runner():
            try:
                result = await coro
            except Exception as e:
                result = e
            if callback and bridge:
                bridge.sig_result.emit(callback, result)

        asyncio.run_coroutine_threadsafe(_runner(), self.loop)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Alice Chat")
    app.setOrganizationName("AliceAI")
    setup_theme(app)

    bridge = _CallbackBridge()
    async_loop = AsyncLoopThread()
    async_loop._bridge = bridge
    async_loop.start()

    session = AgentSession(presets_dir="presets")
    session._async_loop = async_loop.loop  # 供退出时同步保存

    def async_run(coro, callback=None):
        async_loop.run_coro(coro, callback)

    window = MainWindow(session, async_run)
    window.show()

    bridge.sig_result.connect(lambda cb, res: cb(res))

    try:
        from alice.ui.updater import check_update_async
        async_loop.run_coro(check_update_async(window))
    except Exception:
        pass

    exit_code = app.exec()

    async_loop.loop.call_soon_threadsafe(async_loop.loop.stop)
    async_loop.join(timeout=2)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
