"""
ImagePanel — 图片生成调试面板
独立 prompt 输入 + 生图 + 预览，不依赖聊天。
"""
import asyncio
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QLabel, QScrollArea, QProgressBar,
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QPixmap


class ImagePanel(QWidget):
    """图片生成调试面板"""

    def __init__(self, session, async_run, parent=None):
        super().__init__(parent)
        self.session = session
        self.async_run = async_run
        self._gen_thread = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        # 状态
        self.status_label = QLabel("选择图片引擎并连接后即可测试")
        self.status_label.setStyleSheet("color: #888; font-size: 12px;")
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Prompt 输入
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText(
            "输入图片描述...\n例: a cute cat sitting on a sofa, warm lighting, cozy room"
        )
        self.prompt_input.setMaximumHeight(120)
        self.prompt_input.setStyleSheet(
            "QTextEdit { background: #1a1a2e; color: #ddd; border: 1px solid #333;"
            " border-radius: 8px; padding: 10px; font-size: 13px; }"
        )
        layout.addWidget(self.prompt_input)

        # 生成按钮
        btn_row = QHBoxLayout()
        self.generate_btn = QPushButton("🎨 生成图片")
        self.generate_btn.clicked.connect(self._on_generate)
        self.generate_btn.setStyleSheet(
            "QPushButton { background: #7c3aed; color: white; border: none;"
            " border-radius: 8px; padding: 10px 24px; font-size: 14px; }"
            "QPushButton:hover { background: #6d28d9; }"
            "QPushButton:disabled { background: #444; }"
        )
        btn_row.addWidget(self.generate_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # 进度条
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # 图片预览
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(300)
        self.preview.setStyleSheet(
            "QLabel { background: #0a0a14; border: 1px solid #333; border-radius: 8px; }"
        )
        self.preview.setText("图片将在这里显示")
        layout.addWidget(self.preview, stretch=1)

        layout.addStretch()

    def _on_generate(self):
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            return

        adapter = None
        if self.session.agent:
            adapter = self.session.agent.image_gen
        if not adapter:
            self.status_label.setText("❌ 请先在设置面板连接图片引擎")
            self.status_label.setStyleSheet("color: #f55; font-size: 12px;")
            return

        self.generate_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.status_label.setText("⏳ 生成中...")
        self.status_label.setStyleSheet("color: #ccc; font-size: 12px;")
        self.preview.setText("")

        # 后台线程生成
        class GenThread(QThread):
            finished = pyqtSignal(object)

            def __init__(self, adapter, prompt):
                super().__init__()
                self.adapter = adapter
                self.prompt = prompt

            def run(self):
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    path = loop.run_until_complete(
                        self.adapter.generate(self.prompt, "test"))
                    loop.close()
                    self.finished.emit(path)
                except Exception as e:
                    self.finished.emit(e)

        def on_done(result):
            self.generate_btn.setEnabled(True)
            self.progress.setVisible(False)
            if isinstance(result, Exception):
                err_msg = str(result)
                # 同时复制到剪贴板，方便排查
                from PyQt5.QtWidgets import QApplication
                QApplication.clipboard().setText(err_msg)
                self.status_label.setText(f"❌ 生成失败: {err_msg}\n\n(错误信息已自动复制到剪贴板)")
                self.status_label.setStyleSheet("color: #f55; font-size: 12px;")
            else:
                pm = QPixmap(result)
                if pm.isNull():
                    self.status_label.setText("❌ 无法加载图片")
                else:
                    pm = pm.scaled(self.preview.width(), self.preview.height(),
                                   Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.preview.setPixmap(pm)
                    self.status_label.setText(f"✅ 已保存: {result}")
                    self.status_label.setStyleSheet("color: #8f8; font-size: 12px;")

        self._gen_thread = GenThread(adapter, prompt)
        self._gen_thread.finished.connect(on_done)
        self._gen_thread.start()
