"""
MainWindow — 桌面版主窗口
左侧图标导航栏 + 顶部情绪仪表 + 右侧内容面板
"""

import time

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStackedWidget, QPushButton, QLabel, QStatusBar, QMessageBox,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal

from alice.ui.session import AgentSession
from alice.ui.emotion import EmotionBar
from alice.ui.chat import ChatPanel
from alice.ui.settings import SettingsPanel
from alice.ui.workshop import WorkshopPanel
from alice.ui.image_panel import ImagePanel
from alice.ui.reference_panel import ReferencePanel


class MainWindow(QMainWindow):
    """主窗口"""

    sig_autonomous_message = pyqtSignal(str)

    def __init__(self, session: AgentSession, async_run):
        super().__init__()
        self.session = session
        self.async_run = async_run
        self._emotion_ready = False  # 初始化完成前不刷新情绪栏

        self.setWindowTitle("Alice Chat — 积温情绪引擎")
        self.setMinimumSize(960, 680)
        self.resize(1100, 780)

        session.on_status_message = self._on_status_message

        self._setup_ui()
        self._setup_timers()
        self._connect_signals()
        self._init_session()

    def closeEvent(self, event):
        """退出前保存当前状态（对话历史 + 情绪状态）"""
        if self.session and self.session.agent:
            print("[MainWindow] 正在保存状态...")
            try:
                import asyncio
                loop = getattr(self.session, '_async_loop', None)
                if loop and loop.is_running():
                    future = asyncio.run_coroutine_threadsafe(
                        self.session.auto_save(), loop
                    )
                    future.result(timeout=3)
                    print("[MainWindow] 状态已保存")
            except Exception as e:
                print(f"[MainWindow] 退出保存失败: {e}")
        event.accept()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 左侧导航栏
        nav = QWidget()
        nav.setFixedWidth(68)
        nav.setObjectName("sidebar")
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(6, 12, 6, 12)
        nav_layout.setSpacing(2)
        nav_layout.setAlignment(Qt.AlignTop)

        # 导航项: (emoji, label, tooltip)
        nav_items = [
            ("💬", "聊天",   "聊天"),
            ("⚙",  "设置",   "设置"),
            ("🎨", "工坊",   "角色工坊"),
            ("📸", "素材",   "人物素材"),
            ("🖼", "图片",   "图片生成"),
        ]
        self._nav_buttons = []

        for i, (emoji, label, tooltip) in enumerate(nav_items):
            btn = QPushButton(f"{emoji}\n{label}")
            btn.setToolTip(tooltip)
            btn.setCheckable(True)
            btn.setFixedSize(56, 52)
            btn.setObjectName("navBtn")
            if i == 0:
                btn.setChecked(True)
            nav_layout.addWidget(btn, alignment=Qt.AlignHCenter)
            self._nav_buttons.append(btn)

        self.btn_chat = self._nav_buttons[0]
        self.btn_settings = self._nav_buttons[1]
        self.btn_workshop = self._nav_buttons[2]
        self.btn_reference = self._nav_buttons[3]
        self.btn_image = self._nav_buttons[4]

        nav_layout.addStretch()
        root.addWidget(nav)

        # 右侧内容
        content = QVBoxLayout()
        content.setContentsMargins(8, 6, 8, 6)
        content.setSpacing(6)

        self.emotion_bar = EmotionBar()
        content.addWidget(self.emotion_bar)

        self.stack = QStackedWidget()
        self.chat_panel = ChatPanel(self.session, self.async_run)
        self.settings_panel = SettingsPanel(self.session, self.async_run)
        self.workshop_panel = WorkshopPanel(self.session, self.async_run)
        self.reference_panel = ReferencePanel(self.session, self.async_run)
        self.image_panel = ImagePanel(self.session, self.async_run)
        self.stack.addWidget(self.chat_panel)        # 0
        self.stack.addWidget(self.settings_panel)    # 1
        self.stack.addWidget(self.workshop_panel)    # 2
        self.stack.addWidget(self.reference_panel)   # 3
        self.stack.addWidget(self.image_panel)       # 4
        content.addWidget(self.stack, stretch=1)
        root.addLayout(content, stretch=1)

        # 状态栏
        self.status_bar = QStatusBar()
        self.status_label = QLabel("○ 无 LLM")
        self.save_label = QLabel("")
        self.status_bar.addWidget(self.status_label)
        self.status_bar.addPermanentWidget(self.save_label)
        self.setStatusBar(self.status_bar)

    def _setup_timers(self):
        self.timer_emotion = QTimer(self)
        self.timer_emotion.timeout.connect(self._on_emotion_timer)
        self.timer_emotion.start(500)

        self.timer_tick = QTimer(self)
        self.timer_tick.timeout.connect(self._on_tick_timer)
        self.timer_tick.start(3000)

    def _connect_signals(self):
        self.btn_chat.clicked.connect(lambda: self._switch_panel(0, self.btn_chat))
        self.btn_settings.clicked.connect(lambda: self._switch_panel(1, self.btn_settings))
        self.btn_workshop.clicked.connect(lambda: self._switch_panel(2, self.btn_workshop))
        self.btn_reference.clicked.connect(lambda: self._switch_panel(3, self.btn_reference))
        self.btn_image.clicked.connect(lambda: self._switch_panel(4, self.btn_image))
        self.sig_autonomous_message.connect(self.chat_panel.append_autonomous)

        # 设置面板信号
        self.settings_panel.sig_preset_changed.connect(self._on_preset_switched)
        self.settings_panel.sig_llm_changed.connect(self._on_llm_changed)
        self.settings_panel.sig_image_changed.connect(lambda _: self._update_chat_llm_status())

        # 工坊训练完成 → 刷新设置面板 + 素材面板预设列表
        self.workshop_panel.sig_training_done.connect(self._on_training_done)

        # 预设切换 → 刷新素材面板可选列表
        self.settings_panel.sig_preset_changed.connect(
            lambda _: self.reference_panel.refresh_preset_list()
        )

    def _switch_panel(self, idx: int, btn: QPushButton):
        self.stack.setCurrentIndex(idx)
        for b in self._nav_buttons:
            b.setChecked(b is btn)

    def _init_session(self):
        from alice.ui.settings import _load_settings
        from alice.ui.state import restore_chat_history as _restore_chat_history
        from alice.ui.state import restore_emotion_state as _restore_emotion_state
        s = _load_settings()

        # 1. 确定初始人格（上次使用的 或 默认）
        last_preset = s.get("last_preset", "")
        initial_preset = last_preset if last_preset else "默认"

        self.session.create_agent(initial_preset, llm_call=None)
        self.reference_panel.refresh_preset_list()
        self.settings_panel.refresh_preset_list()

        # 确保下拉框显示当前实际人格
        if initial_preset != "默认":
            idx = self.settings_panel.preset_combo.findText(initial_preset)
            if idx >= 0:
                self.settings_panel.preset_combo.setCurrentIndex(idx)

        # 异步初始化: 加载引擎 → 恢复情绪状态 → 恢复对话 → 刷新 UI
        async def _async_init():
            await self.session.ensure_loaded()
            _restore_emotion_state(
                self.session.checkpoint_manager, initial_preset, self.session.agent
            )
            restored = _restore_chat_history(
                self.session.checkpoint_manager, initial_preset
            )
            if restored and self.session.agent:
                self.session.agent.conversation_history = restored
            await self.session.refresh_display_state()
            return restored

        def _on_init_done(restored):
            if restored:
                self.chat_panel.load_history(restored)
            self._emotion_ready = True
            # 触发情绪栏首次刷新
            self._on_emotion_timer()

        self.async_run(_async_init(), _on_init_done)

        # 2. 自动重连 LLM（如有保存的凭据）
        provider = s.get("provider", "")
        api_key = s.get("api_key", "")
        model = s.get("model", "")
        base_url = s.get("base_url", "")

        if provider and provider != "无" and api_key:
            self.status_label.setText(f"⏳ 正在连接 {provider}...")
            def on_llm_restored(result):
                if isinstance(result, Exception):
                    self.status_label.setText(f"⚠️ 自动连接失败: {result}")
                else:
                    self._on_llm_changed(True)
            self.async_run(
                self.session.connect_llm(provider, api_key, model, base_url),
                on_llm_restored,
            )

        # 3. 自动重连图片引擎（异步验证通过后才标记连接）
        img_provider = s.get("img_provider", "")
        img_api_key = s.get("img_api_key", "")
        img_model = s.get("img_model", "")

        if img_provider and img_provider != "无":
            if img_provider == "Diffusers (本地)":
                # Diffusers 需要后台线程加载模型
                from PyQt5.QtCore import QThread, pyqtSignal
                _session = self.session
                _update_status = self._update_chat_llm_status

                class _RestoreDiffusers(QThread):
                    finished_ok = pyqtSignal()

                    def run(self):
                        try:
                            from alice.image.diffusers import DiffusersAdapter
                            adapter = DiffusersAdapter(
                                model=img_model or "models/sd-v1-5",
                                save_dir="images",
                            )
                            adapter.load_model()
                            if adapter._pipe:
                                _session._image_gen_adapter = adapter
                                _session._image_gen_provider = "Diffusers"
                                if _session.agent:
                                    _session.agent.set_image_gen(adapter)
                                self.finished_ok.emit()
                        except Exception as e:
                            print(f"[MainWindow] Diffusers 恢复失败: {e}")

                self._diffusers_restore = _RestoreDiffusers()
                self._diffusers_restore.finished_ok.connect(
                    lambda: (_update_status(), print("[MainWindow] Diffusers 已恢复"))
                )
                self._diffusers_restore.start()
            else:
                async def _restore_image_gen():
                    from alice.image import create_image_gen
                    provider_key = {
                        "Replicate": "replicate",
                        "OpenAI (DALL-E)": "openai",
                        "SD WebUI": "sdwebui",
                    }.get(img_provider, "")
                    if not provider_key:
                        return
                    adapter = create_image_gen(
                        provider_key, api_key=img_api_key or "",
                        model=img_model or "", save_dir="images",
                    )
                    if not adapter:
                        return
                    try:
                        await adapter.validate()
                    except Exception as e:
                        print(f"[MainWindow] 图片引擎验证失败 ({img_provider}): {e}")
                        return
                    self.session._image_gen_adapter = adapter
                    self.session._image_gen_provider = img_provider
                    if self.session.agent:
                        self.session.agent.set_image_gen(adapter)
                    print(f"[MainWindow] 图片引擎已恢复: {img_provider}")
                    self._update_chat_llm_status()

                self.async_run(_restore_image_gen())

        # 注: _emotion_ready 由 _on_init_done 在上方异步初始化完成后设置

    # ---- 定时器 ----

    def _on_emotion_timer(self):
        if not self._emotion_ready:
            return
        state = self.session.last_state
        diag = self.session.last_diag
        self.emotion_bar.update_state(state, diag, self.session.llm_connected,
                                      self.session.preset_name,
                                      self.session.agent.engine._last_activity if self.session.agent else None)
        now = time.time()
        if self.session._last_save_success > 0:
            sec = int(now - self.session._last_save_success)
            self.save_label.setText("已保存" if sec < 3 else "")
        else:
            self.save_label.setText("")

    def _on_tick_timer(self):
        now = time.time()
        self.async_run(self.session.tick(1.0 / 60.0))

        if now - self.session._last_auto_check > 8 and self.session.agent and self.session.agent._llm_call:
            self.session._last_auto_check = now
            def on_auto_check(msg):
                if msg:
                    self.sig_autonomous_message.emit(msg)
            self.async_run(self.session.check_autonomous(), on_auto_check)

        if now - self.session._last_auto_save > 10:
            self.session._last_auto_save = now
            self.async_run(self.session.auto_save())

    # ---- 信号处理 ----

    def _on_status_message(self, msg: str):
        self.status_label.setText(msg)

    def _on_preset_switched(self, preset_name: str):
        self.chat_panel.clear()
        history = self.session.agent.conversation_history if self.session.agent else []
        if history:
            self.chat_panel.load_history(history)
        self.status_label.setText(f"✅ 已切换: {preset_name}")
        self._update_chat_llm_status()
        # 直接读取 switch_preset 已刷新的状态更新情绪栏
        self._on_emotion_timer()

    def _on_llm_changed(self, _connected: bool):
        """LLM 连接/断开 → 同步状态到聊天面板和状态栏"""
        self._update_chat_llm_status()
        if self.session.llm_connected:
            provider = self.session.llm_provider
            model = self.settings_panel.model_input.text().strip()
            info = f"{provider}" + (f" ({model})" if model else "")
            self.status_label.setText(f"● 已连接: {info}")
            self.setWindowTitle(f"Alice Chat — {info}")
        else:
            self.status_label.setText("○ 无 LLM")
            self.setWindowTitle("Alice Chat — 积温情绪引擎")

    def _update_chat_llm_status(self):
        connected = self.session.llm_connected
        provider = self.session.llm_provider
        model = self.settings_panel.model_input.text().strip() if connected else ""
        self.chat_panel.update_llm_status(connected, provider, model)

    def _on_training_done(self, preset_name: str):
        """训练完成后刷新预设列表并自动切换"""
        self.settings_panel.refresh_preset_list()
        self.settings_panel.preset_combo.setCurrentText(preset_name)

    # ---- 更新 ----

    def show_update_notification(self, version: str, download_url: str):
        reply = QMessageBox.question(
            self, "发现新版本",
            f"新版本 {version} 已发布，是否立即更新？\n\n"
            "更新将自动下载并替换当前版本。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if reply == QMessageBox.Yes:
            from alice.ui.updater import download_and_apply_update
            self.async_run(download_and_apply_update(download_url))
