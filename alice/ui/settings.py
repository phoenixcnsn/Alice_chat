"""
SettingsPanel — 设置面板（预设 + LLM + 操控）
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton,
    QLineEdit, QLabel, QSlider, QGroupBox, QScrollArea, QMessageBox,
    QProgressBar, QFileDialog,
)
import json
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal

from alice.ui.session import AgentSession

_SETTINGS_PATH = Path(__file__).parent.parent.parent / "data" / "settings.json"


def _load_settings() -> dict:
    """加载所有持久化设置（含 API Key — 本地桌面应用）"""
    if _SETTINGS_PATH.exists():
        try:
            return json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_settings(provider: str = "", model: str = "", base_url: str = "",
                   img_provider: str = "", img_model: str = "", img_base_url: str = "",
                   api_key: str = "", img_api_key: str = "", last_preset: str = ""):
    """保存设置到文件"""
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # 增量更新：读取已有数据，只覆盖传入的非空字段
    existing = {}
    if _SETTINGS_PATH.exists():
        try:
            existing = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    updates = {
        "provider": provider, "model": model, "base_url": base_url,
        "img_provider": img_provider, "img_model": img_model, "img_base_url": img_base_url,
        "api_key": api_key, "img_api_key": img_api_key, "last_preset": last_preset,
    }
    # 只覆盖传入的非空值，保留已有数据
    for k, v in updates.items():
        if v:
            existing[k] = v
        elif k not in existing:
            existing.setdefault(k, "")

    _SETTINGS_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


class SettingsPanel(QWidget):
    """设置面板"""

    sig_preset_changed = pyqtSignal(str)   # 预设切换通知
    sig_llm_changed = pyqtSignal(bool)     # LLM 连接状态变化
    sig_image_changed = pyqtSignal(bool)   # 图片引擎连接状态变化

    def __init__(self, session: AgentSession, async_run, parent=None):
        super().__init__(parent)
        self.session = session
        self.async_run = async_run
        self._setup_ui()

    def _setup_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(12)

        # ---- 人格预设 ----
        preset_group = QGroupBox("🎭 人格预设")
        preset_layout = QVBoxLayout(preset_group)

        row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.currentTextChanged.connect(self._on_preset_selected)
        row.addWidget(self.preset_combo, stretch=1)

        self.delete_btn = QPushButton("🗑 删除")
        self.delete_btn.setVisible(False)
        self.delete_btn.clicked.connect(self._on_delete_preset)
        row.addWidget(self.delete_btn)
        preset_layout.addLayout(row)
        layout.addWidget(preset_group)

        # ---- LLM 配置 ----
        llm_group = QGroupBox("🔌 LLM API")
        llm_layout = QVBoxLayout(llm_group)

        llm_layout.addWidget(QLabel("提供商"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["DeepSeek", "Anthropic", "OpenAI", "无"])
        llm_layout.addWidget(self.provider_combo)

        llm_layout.addWidget(QLabel("API Key"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("sk-...")
        llm_layout.addWidget(self.api_key_input)

        llm_layout.addWidget(QLabel("Base URL（留空使用默认）"))
        self.base_url_input = QLineEdit()
        self.base_url_input.setPlaceholderText("https://api.deepseek.com/v1")
        llm_layout.addWidget(self.base_url_input)

        llm_layout.addWidget(QLabel("Model（留空使用默认）"))
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("deepseek-chat / claude-sonnet-4-6 / gpt-4o")
        llm_layout.addWidget(self.model_input)

        btn_row = QHBoxLayout()
        self.connect_btn = QPushButton("🔗 连接 LLM")
        self.connect_btn.clicked.connect(self._on_connect)
        self.disconnect_btn = QPushButton("🔌 断开")
        self.disconnect_btn.clicked.connect(self._on_disconnect)
        btn_row.addWidget(self.connect_btn)
        btn_row.addWidget(self.disconnect_btn)
        llm_layout.addLayout(btn_row)

        layout.addWidget(llm_group)

        # ---- 图片生成配置 ----
        img_group = QGroupBox("🖼 图片生成")
        img_layout = QVBoxLayout(img_group)

        img_layout.addWidget(QLabel("引擎"))
        self.img_provider_combo = QComboBox()
        self.img_provider_combo.addItems(["无", "Replicate", "OpenAI (DALL-E)", "SD WebUI", "Diffusers (本地)"])
        self.img_provider_combo.currentTextChanged.connect(self._on_img_provider_changed)
        img_layout.addWidget(self.img_provider_combo)

        img_layout.addWidget(QLabel("API Key"))
        self.img_api_key = QLineEdit()
        self.img_api_key.setEchoMode(QLineEdit.Password)
        self.img_api_key.setPlaceholderText("r8_... 或 sk-...")
        img_layout.addWidget(self.img_api_key)

        img_layout.addWidget(QLabel("Base URL / Model"))
        img_url_row = QHBoxLayout()
        self.img_base_url = QLineEdit()
        self.img_base_url.setPlaceholderText("http://localhost:7860 或 dall-e-3")
        img_url_row.addWidget(self.img_base_url, stretch=1)
        self.img_browse_btn = QPushButton("📁 浏览")
        self.img_browse_btn.setVisible(False)
        self.img_browse_btn.clicked.connect(self._on_browse_model_folder)
        img_url_row.addWidget(self.img_browse_btn)
        img_layout.addLayout(img_url_row)

        self.img_status = QLabel("⚪ 未启用")
        self.img_status.setStyleSheet("color: #888; font-size: 11px;")
        img_layout.addWidget(self.img_status)

        self.img_progress = QProgressBar()
        self.img_progress.setVisible(False)
        self.img_progress.setRange(0, 100)
        img_layout.addWidget(self.img_progress)

        self.img_connect_btn = QPushButton("🔗 连接图片引擎")
        self.img_connect_btn.clicked.connect(self._on_connect_image)
        img_layout.addWidget(self.img_connect_btn)

        layout.addWidget(img_group)

        # ---- 操控 ----

        # ---- 操控 ----
        ctrl_group = QGroupBox("🎮 操控")
        ctrl_layout = QVBoxLayout(ctrl_group)

        # 情绪事件
        ctrl_layout.addWidget(QLabel("情绪事件"))
        event_row = QHBoxLayout()
        self.event_axis = QComboBox()
        self.event_axis.addItems(['connection', 'pride', 'valence', 'arousal', 'mood', 'immersion'])
        self.event_value = QSlider(Qt.Horizontal)
        self.event_value.setRange(-100, 100)
        self.event_value.setValue(0)
        self.event_btn = QPushButton("⚡ 施加")
        self.event_btn.clicked.connect(self._on_apply_event)
        event_row.addWidget(self.event_axis, 2)
        event_row.addWidget(self.event_value, 3)
        event_row.addWidget(self.event_btn, 1)
        ctrl_layout.addLayout(event_row)

        # 活动
        ctrl_layout.addWidget(QLabel("活动"))
        act_row = QHBoxLayout()
        self.activity_combo = QComboBox()
        self.activity_combo.addItems(['reading', 'search', 'browse', 'observe'])
        self.activity_label = QLineEdit()
        self.activity_label.setPlaceholderText("活动描述（可选）")
        self.activity_btn = QPushButton("🎯 设置")
        self.activity_btn.clicked.connect(self._on_set_activity)
        act_row.addWidget(self.activity_combo, 2)
        act_row.addWidget(self.activity_label, 3)
        act_row.addWidget(self.activity_btn, 1)
        ctrl_layout.addLayout(act_row)

        # 等待
        ctrl_layout.addWidget(QLabel("时间快进（分钟）"))
        wait_row = QHBoxLayout()
        self.wait_slider = QSlider(Qt.Horizontal)
        self.wait_slider.setRange(5, 1200)
        self.wait_slider.setValue(60)
        self.wait_btn = QPushButton("⏩ 快进")
        self.wait_btn.clicked.connect(self._on_wait)
        wait_row.addWidget(self.wait_slider, 4)
        wait_row.addWidget(self.wait_btn, 1)
        ctrl_layout.addLayout(wait_row)

        # 重置
        reset_row = QHBoxLayout()
        self.reset_conn_btn = QPushButton("💔 重置思念")
        self.reset_conn_btn.clicked.connect(self._on_reset_connection)
        self.hard_reset_btn = QPushButton("🔄 硬重置")
        self.hard_reset_btn.clicked.connect(self._on_hard_reset)
        reset_row.addWidget(self.reset_conn_btn)
        reset_row.addWidget(self.hard_reset_btn)
        ctrl_layout.addLayout(reset_row)

        layout.addWidget(ctrl_group)
        layout.addStretch()

        scroll.setWidget(w)

        # 恢复上次保存的设置
        s = _load_settings()
        if s.get("provider"):
            idx = self.provider_combo.findText(s["provider"])
            if idx >= 0:
                self.provider_combo.setCurrentIndex(idx)
        if s.get("model"):
            self.model_input.setText(s["model"])
        if s.get("base_url"):
            self.base_url_input.setText(s["base_url"])
        if s.get("api_key"):
            self.api_key_input.setText(s["api_key"])
        if s.get("img_provider"):
            idx = self.img_provider_combo.findText(s["img_provider"])
            if idx >= 0:
                self.img_provider_combo.setCurrentIndex(idx)
        if s.get("img_model"):
            self.img_base_url.setText(s["img_model"])
        if s.get("img_api_key"):
            self.img_api_key.setText(s["img_api_key"])

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # 统一样式
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #ccc;
                border: 1px solid #333;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 16px;
            }
            QGroupBox::title { subcontrol-origin: margin; padding: 0 6px; }
            QComboBox, QLineEdit {
                background: #1a1a2e; color: #ddd; border: 1px solid #333;
                border-radius: 6px; padding: 6px;
            }
            QComboBox::drop-down { border: none; }
            QPushButton {
                background: #2a2a3a; color: #ddd; border: 1px solid #444;
                border-radius: 6px; padding: 6px 12px;
            }
            QPushButton:hover { background: #3a3a4a; }
            QSlider::groove:horizontal { height: 6px; background: #333; border-radius: 3px; }
            QSlider::handle:horizontal { background: #7c3aed; width: 14px; border-radius: 7px; }
        """)

    # ---- 公共 API ----

    def refresh_preset_list(self):
        """刷新预设下拉列表"""
        choices = self.session.preset_manager.list_all()
        current = self.preset_combo.currentText()
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItems(choices if choices else ["默认"])
        if current in choices:
            self.preset_combo.setCurrentText(current)
        self.preset_combo.blockSignals(False)
        self.delete_btn.setVisible(self.preset_combo.currentText() != "默认")

    # ---- 事件处理 ----

    def _on_preset_selected(self, preset: str):
        """下拉选择预设 → 确认切换"""
        if preset == self.session.preset_name:
            return
        reply = QMessageBox.question(
            self, "切换人格",
            f'切换人格预设到 "{preset}" 吗？\n\n'
            "当前对话和情绪状态将被保存，新预设的历史将被加载。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply == QMessageBox.Yes:
            def on_done(result):
                self.refresh_preset_list()
                _save_settings(last_preset=preset)
                self.sig_preset_changed.emit(preset)

            self.async_run(self.session.switch_preset(preset), on_done)
        else:
            # 回退下拉框
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText(self.session.preset_name)
            self.preset_combo.blockSignals(False)

    def _on_delete_preset(self):
        """删除预设"""
        preset = self.preset_combo.currentText()
        if preset == "默认":
            return
        reply = QMessageBox.warning(
            self, "删除预设",
            f'确定删除预设 "{preset}" 吗？此操作不可撤销！',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.session.preset_manager.delete(preset)
            self.refresh_preset_list()
            if self.session.preset_name == preset:
                def on_done(r):
                    self.refresh_preset_list()
                    self.sig_preset_changed.emit("默认")

                self.async_run(self.session.switch_preset("默认"), on_done)

    def _on_connect(self):
        """连接 LLM（含 API Key 验证）"""
        provider = self.provider_combo.currentText()
        if provider == "无":
            self._on_disconnect()
            return
        api_key = self.api_key_input.text().strip()
        model = self.model_input.text().strip()
        base_url = self.base_url_input.text().strip()

        if not api_key:
            QMessageBox.warning(self, "提示", "请输入 API Key")
            return

        # UI 进入验证状态
        self.connect_btn.setEnabled(False)
        self.connect_btn.setText("⏳ 验证中...")

        def on_done(result):
            self.connect_btn.setEnabled(True)
            self.connect_btn.setText("🔗 连接 LLM")
            if isinstance(result, Exception):
                QMessageBox.critical(self, "LLM 连接失败", str(result))
                return
            success = result
            self.session.llm_connected = success
            self.session.llm_provider = provider if success else "无"
            if success:
                self.sig_llm_changed.emit(True)
                _save_settings(provider=provider, model=model, base_url=base_url,
                              api_key=api_key)
            # 失败已在 connect_llm 中抛出异常，这里不会走到 success=False

        self.async_run(
            self.session.connect_llm(provider, api_key, model, base_url),
            on_done,
        )

    def _on_disconnect(self):
        self.session.disconnect_llm()
        self.sig_llm_changed.emit(False)

    def _on_apply_event(self):
        axis = self.event_axis.currentText()
        val = self.event_value.value() / 100.0
        delta = {}
        if axis == 'mood':
            delta['valence'] = val
            delta['arousal'] = val * 0.5
        else:
            delta[axis] = val
        agent = self.session.agent
        if agent:
            self.async_run(agent.apply_event(delta))
            self.async_run(self.session.refresh_display_state())

    def _on_set_activity(self):
        act = self.activity_combo.currentText()
        label = self.activity_label.text().strip() or act
        agent = self.session.agent
        if agent:
            self.async_run(agent.set_activity(act, label))
            self.async_run(self.session.refresh_display_state())

    def _on_wait(self):
        minutes = self.wait_slider.value() / 10.0
        agent = self.session.agent
        if agent:
            self.async_run(agent.tick_time(minutes))
            self.async_run(self.session.refresh_display_state())

    def _on_reset_connection(self):
        agent = self.session.agent
        if agent:
            self.async_run(agent.reset_connection())
            self.async_run(self.session.refresh_display_state())

    # ---- 图片生成 ----

    def _on_img_provider_changed(self, provider: str):
        """切换图片引擎时更新 UI"""
        show_key = provider in ("Replicate", "OpenAI (DALL-E)")
        show_url = provider in ("SD WebUI", "Diffusers (本地)")
        self.img_api_key.setVisible(show_key)
        self.img_browse_btn.setVisible(provider == "Diffusers (本地)")
        self.img_base_url.setVisible(True)
        if provider == "SD WebUI":
            self.img_base_url.setPlaceholderText("http://localhost:7860")
        elif provider == "Replicate":
            self.img_base_url.setPlaceholderText("black-forest-labs/flux-2-pro")
        elif provider == "OpenAI (DALL-E)":
            self.img_base_url.setPlaceholderText("dall-e-3")
        elif provider == "Diffusers (本地)":
            self.img_base_url.setPlaceholderText("alice/model/AI-ModelScope/stable-diffusion-v1-5")
        else:
            self.img_base_url.setVisible(False)

    def _on_browse_model_folder(self):
        """选择本地模型文件夹"""
        path = QFileDialog.getExistingDirectory(self, "选择 SD 模型文件夹")
        if path:
            self.img_base_url.setText(path)

    def _on_connect_image(self):
        """连接图片生成引擎（含验证）"""
        provider = self.img_provider_combo.currentText()
        if provider == "无":
            self.session.disconnect_image_gen()
            self.img_status.setText("⚪ 未启用")
            self.sig_image_changed.emit(False)
            return

        api_key = self.img_api_key.text().strip()
        base_url = self.img_base_url.text().strip()

        if provider == "Replicate" and not api_key:
            QMessageBox.warning(self, "提示", "请输入 Replicate API Key")
            return
        if provider == "OpenAI (DALL-E)" and not api_key:
            QMessageBox.warning(self, "提示", "请输入 OpenAI API Key")
            return

        # Diffusers 走异步加载，不在这里验证
        if provider == "Diffusers (本地)":
            path = base_url or "alice/model/AI-ModelScope/stable-diffusion-v1-5"
            self._connect_diffusers_with_progress(path)
            return

        # 先创建适配器
        try:
            if provider == "Replicate":
                self.session.connect_image_gen("replicate", api_key=api_key, model=base_url)
            elif provider == "OpenAI (DALL-E)":
                self.session.connect_image_gen("openai", api_key=api_key, model=base_url)
            elif provider == "SD WebUI":
                self.session.connect_image_gen("sdwebui", base_url=base_url or "http://localhost:7860")
        except Exception as e:
            QMessageBox.critical(self, "连接失败", str(e))
            self.img_status.setText("❌ 连接失败")
            return

        # 异步验证
        self.img_status.setText("⏳ 验证中...")
        self.img_connect_btn.setEnabled(False)

        async def do_validate():
            adapter = self.session._image_gen_adapter
            if adapter:
                try:
                    await adapter.validate()
                except Exception as e:
                    return e
            return None

        def on_validated(result):
            self.img_connect_btn.setEnabled(True)
            if result is None:
                self.img_status.setText(f"🟢 已连接: {provider}")
                _save_settings(img_provider=provider, img_model=base_url,
                              img_api_key=api_key)
                self.sig_image_changed.emit(True)
            else:
                self.session.disconnect_image_gen()
                QMessageBox.critical(self, "连接失败", str(result))
                self.img_status.setText(f"❌ 验证失败: {result}")
                self.sig_image_changed.emit(False)

        self.async_run(do_validate(), on_validated)

    def _connect_diffusers_with_progress(self, model: str):
        """连接本地 Diffusers，从文件夹加载模型"""
        self.img_status.setText("⏳ 正在加载模型...")
        self.img_progress.setVisible(True)
        self.img_progress.setRange(0, 0)
        self.img_connect_btn.setEnabled(False)

        from PyQt5.QtCore import QThread, pyqtSignal as _pyqtSignal

        class LoadThread(QThread):
            finished = _pyqtSignal(object)
            progress = _pyqtSignal(str, float, float)

            def __init__(self, model_path):
                super().__init__()
                self._path = model_path

            def run(self):
                try:
                    from alice.image.diffusers import DiffusersAdapter
                    adapter = DiffusersAdapter(
                        model=self._path, save_dir="images",
                        progress_callback=lambda t, d, tot: self.progress.emit(t, d, tot),
                    )
                    adapter.load_model()
                    self.finished.emit(adapter)
                except Exception as e:
                    self.finished.emit(e)

        self._load_thread = LoadThread(model)

        def on_progress(text: str, done: float, total: float):
            if total > 0 and self.img_progress.maximum() == 0:
                self.img_progress.setRange(0, 100)
            if total > 0:
                self.img_progress.setValue(int(done / total * 100))
            self.img_status.setText(f"⏳ {text}")

        def on_loaded(adapter):
            self.img_progress.setVisible(False)
            self.img_connect_btn.setEnabled(True)
            if isinstance(adapter, Exception):
                err_msg = str(adapter)
                if len(err_msg) > 300:
                    err_msg = err_msg[:300] + "..."
                self.img_status.setText(f"❌ {err_msg}")
                self.sig_image_changed.emit(False)
                return
            if adapter and adapter._pipe:
                self.session._image_gen_adapter = adapter
                self.session._image_gen_provider = "Diffusers"
                if self.session.agent:
                    self.session.agent.set_image_gen(adapter)
                self.img_status.setText("🟢 已连接: Diffusers (本地)")
                _save_settings(img_provider="Diffusers (本地)", img_model=model)
                self.sig_image_changed.emit(True)
            else:
                self.img_status.setText("❌ 模型加载失败")

        self._load_thread.progress.connect(on_progress)
        self._load_thread.finished.connect(on_loaded)
        self._load_thread.start()

    def _on_hard_reset(self):
        reply = QMessageBox.warning(
            self, "硬重置",
            "这将清空所有对话历史和情绪状态，确定继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            agent = self.session.agent
            if agent:
                self.async_run(agent.reset(hard=True))
                self.async_run(self.session.refresh_display_state())
