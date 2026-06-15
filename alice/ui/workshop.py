"""
WorkshopPanel — 角色工坊面板
支持首次提取和增量训练。
"""

import os

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLineEdit, QPushButton, QTextEdit, QLabel, QProgressBar,
    QFileDialog, QScrollArea, QGroupBox,
)
from PyQt5.QtCore import Qt, pyqtSignal

from alice.ui.session import AgentSession


class WorkshopPanel(QWidget):
    """角色工坊"""

    sig_training_done = pyqtSignal(str)  # preset_name

    def __init__(self, session: AgentSession, async_run, parent=None):
        super().__init__(parent)
        self.session = session
        self.async_run = async_run
        self._setup_ui()

    def _setup_ui(self):
        tabs = QTabWidget()

        # ---- Tab 1: 首次训练 ----
        tab1 = QScrollArea()
        tab1.setWidgetResizable(True)
        t1 = QWidget()
        t1_layout = QVBoxLayout(t1)
        t1_layout.setSpacing(10)

        t1_layout.addWidget(QLabel("角色名"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("为新角色命名...")
        t1_layout.addWidget(self.name_input)

        t1_layout.addWidget(QLabel("源文本（粘贴或上传文件）"))
        self.source_text = QTextEdit()
        self.source_text.setPlaceholderText("在此直接粘贴角色对话/背景文本...")
        self.source_text.setMinimumHeight(200)
        t1_layout.addWidget(self.source_text)

        file_row = QHBoxLayout()
        btn_upload = QPushButton("📁 上传文件")
        btn_upload.clicked.connect(self._on_upload_files)
        file_row.addWidget(btn_upload)
        self.file_label = QLabel("")
        file_row.addWidget(self.file_label, stretch=1)
        t1_layout.addLayout(file_row)

        self.extract_btn = QPushButton("🚀 多阶段提取")
        self.extract_btn.clicked.connect(self._on_extract)
        t1_layout.addWidget(self.extract_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        t1_layout.addWidget(self.progress_bar)

        self.preview_area = QTextEdit()
        self.preview_area.setReadOnly(True)
        self.preview_area.setPlaceholderText("提取结果将在这里预览...")
        t1_layout.addWidget(self.preview_area)

        save_row = QHBoxLayout()
        self.save_btn = QPushButton("💾 保存角色")
        self.save_btn.setVisible(False)
        self.save_btn.clicked.connect(self._on_save)
        save_row.addWidget(self.save_btn)
        self.reset_btn = QPushButton("🔄 重置")
        self.reset_btn.setVisible(False)
        self.reset_btn.clicked.connect(self._on_reset)
        save_row.addWidget(self.reset_btn)
        t1_layout.addLayout(save_row)

        t1_layout.addStretch()
        tab1.setWidget(t1)
        tabs.addTab(tab1, "首次训练")

        # ---- Tab 2: 增量训练 ----
        tab2 = QScrollArea()
        tab2.setWidgetResizable(True)
        t2 = QWidget()
        t2_layout = QVBoxLayout(t2)
        t2_layout.setSpacing(10)

        t2_layout.addWidget(QLabel("已有角色名"))
        self.inc_name_input = QLineEdit()
        self.inc_name_input.setPlaceholderText("要增量训练的角色名...")
        t2_layout.addWidget(self.inc_name_input)

        t2_layout.addWidget(QLabel("新增源文本"))
        self.inc_text = QTextEdit()
        self.inc_text.setPlaceholderText("新增的对话/背景文本...")
        self.inc_text.setMinimumHeight(150)
        t2_layout.addWidget(self.inc_text)

        inc_file_row = QHBoxLayout()
        btn_inc_upload = QPushButton("📁 上传文件")
        btn_inc_upload.clicked.connect(self._on_inc_upload)
        inc_file_row.addWidget(btn_inc_upload)
        self.inc_file_label = QLabel("")
        inc_file_row.addWidget(self.inc_file_label, stretch=1)
        t2_layout.addLayout(inc_file_row)

        self.inc_train_btn = QPushButton("🔧 增量训练")
        self.inc_train_btn.clicked.connect(self._on_inc_train)
        t2_layout.addWidget(self.inc_train_btn)

        self.inc_progress = QProgressBar()
        self.inc_progress.setVisible(False)
        t2_layout.addWidget(self.inc_progress)

        self.inc_preview = QTextEdit()
        self.inc_preview.setReadOnly(True)
        self.inc_preview.setPlaceholderText("训练结果...")
        t2_layout.addWidget(self.inc_preview)

        t2_layout.addStretch()
        tab2.setWidget(t2)
        tabs.addTab(tab2, "增量训练")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(tabs)

        self.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #333; background: #0f0f1a; }
            QTabBar::tab { padding: 8px 16px; color: #888; }
            QTabBar::tab:selected { color: #a78bfa; border-bottom: 2px solid #7c3aed; }
            QTextEdit, QLineEdit {
                background: #1a1a2e; color: #ddd; border: 1px solid #333;
                border-radius: 6px; padding: 6px;
            }
            QPushButton {
                background: #2a2a3a; color: #ddd; border: 1px solid #444;
                border-radius: 6px; padding: 6px 12px;
            }
            QPushButton:hover { background: #3a3a4a; }
            QProgressBar {
                border: 1px solid #333; border-radius: 4px; text-align: center;
                background: #1a1a2e;
            }
            QProgressBar::chunk { background: #7c3aed; border-radius: 3px; }
        """)

        # 保存文件路径
        self._uploaded_files = []
        self._inc_uploaded_files = []

    # ---- 事件处理 ----

    def _on_upload_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择源文本文件", "",
            "Text Files (*.txt *.md);;All Files (*)",
        )
        if files:
            self._uploaded_files = files
            self.file_label.setText(f"已选择 {len(files)} 个文件")
            # 读取文件内容到文本框
            for f in files:
                try:
                    with open(f, encoding='utf-8') as fp:
                        content = fp.read()
                        self.source_text.append(f"--- {os.path.basename(f)} ---\n{content}\n")
                except Exception:
                    pass

    def _on_extract(self):
        """首次训练"""
        name = self.name_input.text().strip()
        text = self.source_text.toPlainText().strip()

        if not text:
            return     # 没有文本不做任何事
        if not name:
            name = "新角色"  # 默认名，训练后 LLM 会提取真实名字

        import time as _time
        self._train_start = _time.time()
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.preview_area.setPlainText("⏳ 训练中，请稍候...")
        self.extract_btn.setEnabled(False)

        if not hasattr(self, '_train_timer'):
            from PyQt5.QtCore import QTimer
            self._train_timer = QTimer(self)
            self._train_timer.timeout.connect(self._update_train_progress)
        self._train_timer.start(1000)

        trainer = self.session.get_or_create_trainer()

        def on_done(result):
            self._train_timer.stop()
            self.progress_bar.setVisible(False)
            self.extract_btn.setEnabled(True)
            elapsed = _time.time() - self._train_start
            if isinstance(result, Exception):
                self.preview_area.setHtml(
                    f"<span style='color:red'>❌ 训练失败 (耗时 {elapsed:.0f}s): {result}</span>")
                return
            profile = result
            if hasattr(profile, 'to_dict'):
                import json
                # 自动填入 LLM 提取的角色名
                extracted_name = profile.name
                if extracted_name and extracted_name != name and self.name_input.text().strip() == name:
                    self.name_input.setText(extracted_name)
                preview = json.dumps(profile.to_dict(), ensure_ascii=False, indent=2)
                self.preview_area.setPlainText(
                    f"# 训练完成 (耗时 {elapsed:.0f}s)\n\n{preview}")
                self.save_btn.setVisible(True)
                self.reset_btn.setVisible(True)
                self.session._extracted_profile = profile

        texts = [text] if text else None
        files = self._uploaded_files if self._uploaded_files else None
        self.async_run(trainer.train(texts=texts, files=files, name_hint=name), on_done)

    def _update_train_progress(self):
        """更新训练进度提示"""
        import time as _time
        elapsed = int(_time.time() - self._train_start)
        self.preview_area.setPlainText(f"⏳ 训练中... 已耗时 {elapsed}s")

    def _on_save(self):
        """保存提取的角色"""
        profile = self.session._extracted_profile
        if not profile:
            return

        new_name = self.name_input.text().strip()
        if new_name:
            profile.name = new_name

        self.session.preset_manager.save(profile)
        self.session.checkpoint_manager.save_profile(profile)
        self.session._extracted_profile = None
        self.save_btn.setVisible(False)
        self.reset_btn.setVisible(False)
        self.preview_area.setHtml(f"<span style='color:#8f8'>✅ 已保存: {profile.name}</span>")
        self.sig_training_done.emit(profile.name)

        # 清空文本，方便训练下一个角色
        self.source_text.clear()
        self.file_label.clear()
        self._uploaded_files.clear()

    def _on_reset(self):
        """重置工坊"""
        self.name_input.clear()
        self.source_text.clear()
        self.preview_area.clear()
        self.file_label.clear()
        self._uploaded_files.clear()
        self.session._extracted_profile = None
        self.save_btn.setVisible(False)
        self.reset_btn.setVisible(False)
        self.progress_bar.setVisible(False)

    def _on_inc_upload(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择新增文本文件", "",
            "Text Files (*.txt *.md);;All Files (*)",
        )
        if files:
            self._inc_uploaded_files = files
            self.inc_file_label.setText(f"已选择 {len(files)} 个文件")
            for f in files:
                try:
                    with open(f, encoding='utf-8') as fp:
                        self.inc_text.append(fp.read() + "\n")
                except Exception:
                    pass

    def _on_inc_train(self):
        """增量训练"""
        name = self.inc_name_input.text().strip()
        text = self.inc_text.toPlainText().strip()
        if not name or not text:
            return

        self.inc_progress.setVisible(True)
        self.inc_progress.setRange(0, 0)

        trainer = self.session.get_or_create_trainer()

        def on_done(result):
            self.inc_progress.setVisible(False)
            if isinstance(result, Exception):
                self.inc_preview.setHtml(f"<span style='color:red'>❌ {result}</span>")
                return
            summary = trainer.get_checkpoint_summary(name)
            self.inc_preview.setHtml(
                f"<span style='color:#8f8'>✅ 增量训练完成！</span><br>"
                f"累计语料: {summary.get('corpus_chars', 0)} 字符<br>"
                f"训练批次: {summary.get('version', 1)}"
            )
            self.sig_training_done.emit(name)

        texts = [text] if text else None
        files = self._inc_uploaded_files if self._inc_uploaded_files else None
        self.async_run(
            trainer.train_incremental(name, texts=texts, files=files),
            on_done,
        )
