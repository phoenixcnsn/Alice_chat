"""
诊断脚本 — 逐步测试每个组件，定位崩溃点。
运行: python alice_app/debug_test.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtCore import Qt

app = QApplication(sys.argv)

# Step 1: 基本窗口
print("[1/7] Basic QMainWindow...", flush=True)
w = QMainWindow()
w.setWindowTitle("debug")
w.resize(400, 200)
label = QLabel("test label")
w.setCentralWidget(label)
print("[1/7] OK", flush=True)

# Step 2: Fusion 主题
print("[2/7] Fusion theme...", flush=True)
app.setStyle("Fusion")
palette = app.palette()
palette.setColor(palette.Window, Qt.black)
palette.setColor(palette.WindowText, Qt.white)
app.setPalette(palette)
print("[2/7] OK", flush=True)

# Step 3: QSS 样式表
print("[3/7] QSS stylesheet...", flush=True)
qss_path = Path(__file__).parent / "resources" / "style.qss"
if qss_path.exists():
    app.setStyleSheet(qss_path.read_text(encoding="utf-8"))
    print("[3/7] OK", flush=True)
else:
    print("[3/7] Skipped (no style.qss)", flush=True)

# Step 4: AppSession
print("[4/7] AgentSession...", flush=True)
from alice.ui.session import AgentSession
session = AgentSession(presets_dir="presets")
session.create_agent("默认", llm_call=None)
print("[4/7] OK", flush=True)

# Step 5: EmotionBar widget
print("[5/7] EmotionBar...", flush=True)
from alice.ui.emotion import EmotionBar
bar = EmotionBar()
bar.update_state({}, {}, False, "test", None)
print("[5/7] OK", flush=True)

# Step 6: ChatPanel
print("[6/7] ChatPanel...", flush=True)
from alice.ui.chat import ChatPanel
chat = ChatPanel(session, lambda coro, cb=None: None)
print("[6/7] OK", flush=True)

# Step 7: MainWindow
print("[7/7] MainWindow...", flush=True)
from alice.ui.window import MainWindow
window = MainWindow(session, lambda coro, cb=None: None)
print("[7/7] OK", flush=True)

print("\n=== All steps passed! Showing window... ===", flush=True)
window.show()
print("Window shown. Entering event loop...", flush=True)
sys.exit(app.exec())
