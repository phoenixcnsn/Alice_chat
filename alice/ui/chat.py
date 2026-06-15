"""
ChatPanel — 聊天气泡面板
QListView + 自定义 Delegate + 输入区域
支持文字鼠标选中、右键复制
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
    QListView, QLabel, QMenu, QAction, QApplication,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPoint
from PyQt5.QtGui import QTextDocument, QFont

from alice.ui.session import AgentSession
from alice.ui.chat_model import ChatMessageModel
from alice.ui.chat_bubble import ChatBubbleDelegate


# ============================================================
# 支持文字选中的 QListView 子类
# ============================================================
class _SelectableListView(QListView):
    """在 ChatBubbleDelegate 渲染的气泡内支持鼠标拖选文字"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._text_sel = None          # (row, start, end) or None
        self._sel_doc = QTextDocument()  # 用于 hit-test 的临时 document
        self._delegate_ref = None  # type: ChatBubbleDelegate | None

    def set_delegate(self, delegate: ChatBubbleDelegate):
        self._delegate_ref = delegate

    # ---- 文字选择 ----
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._delegate_ref:
            hit = self._hit_test_text(event.pos())
            if hit:
                row, char_pos = hit
                self._text_sel = (row, char_pos, char_pos)
                self.viewport().update()
                self.setFocus()  # 确保障 Ctrl+C 等键盘操作可用
                event.accept()
                return
        # 非左键 / 未命中文字 → 清除选择
        self._clear_sel()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._text_sel and self._delegate_ref:
            row, start, _ = self._text_sel
            hit = self._hit_test_text(event.pos())
            if hit and hit[0] == row:
                self._text_sel = (row, start, hit[1])
                self.viewport().update()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._text_sel:
            sel = self._text_sel
            if sel[1] == sel[2]:  # 没选中任何东西
                self._clear_sel()
            # 保留选择，等 Ctrl+C
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        """双击选中整条消息"""
        if event.button() == Qt.LeftButton and self._delegate_ref:
            hit = self._hit_test_text(event.pos())
            if hit:
                row = hit[0]
                text = self._row_text(row)
                if text:
                    self._text_sel = (row, 0, len(text))
                    self.viewport().update()
                    return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        """Ctrl+C 复制选中文字"""
        if event.key() == Qt.Key_C and event.modifiers() & Qt.ControlModifier:
            if self._text_sel:
                _, start, end = self._text_sel
                if start != end:
                    text = self._row_text(self._text_sel[0])
                    if text:
                        sel_text = text[min(start, end):max(start, end)]
                        QApplication.clipboard().setText(sel_text)
                        self._clear_sel()
                        return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        """右键菜单：复制"""
        menu = QMenu(self)

        # 如果有选中文字 → "复制选中"
        if self._text_sel:
            _, start, end = self._text_sel
            if start != end:
                act_copy_sel = QAction("📋 复制选中文字", self)
                act_copy_sel.triggered.connect(self._copy_selected)
                menu.addAction(act_copy_sel)

        # "复制整条消息"
        hit = self._hit_test_text(event.pos()) if self._delegate_ref else None
        row = hit[0] if hit else None
        act_copy_all = QAction("📋 复制整条消息", self)
        act_copy_all.triggered.connect(lambda: self._copy_full_message(row))
        menu.addAction(act_copy_all)

        if not self._text_sel and row is None:
            act_copy_all.setEnabled(False)

        menu.exec_(event.globalPos())

    # ---- 内部辅助 ----
    def _hit_test_text(self, viewport_pos: QPoint):
        """返回 (row, char_pos) 或 None。
        viewport_pos 是相对于 viewport 的坐标（来自鼠标事件）。
        """
        if not self._delegate_ref:
            return None
        index = self.indexAt(viewport_pos)
        if not index.isValid():
            return None
        row = index.row()

        # 将 viewport 坐标转换为 item 本地坐标（与 delegate paint 一致）
        item_rect = self.visualRect(index)
        local_pos = QPoint(
            viewport_pos.x() - item_rect.x(),
            viewport_pos.y() - item_rect.y(),
        )

        tr = self._delegate_ref.text_rect_for_row(row)
        if tr.isNull() or not tr.contains(local_pos):
            return None

        # 重新布局文档（与 delegate 中一致）
        text = self._row_text(row)
        if not text:
            return None
        doc = self._sel_doc
        doc.setDefaultFont(QFont("Microsoft YaHei", 10))
        doc.setPlainText(text)
        doc.setTextWidth(tr.width())

        # hit-test 文字区域内的字符位置
        char_x = local_pos.x() - tr.x()
        char_y = local_pos.y() - tr.y()
        char_pos = doc.documentLayout().hitTest(
            QPoint(int(char_x), int(char_y)), Qt.ExactHit
        )
        return (row, char_pos)

    def _row_text(self, row: int) -> str:
        if not self.model():
            return ""
        idx = self.model().index(row)
        return idx.data(Qt.DisplayRole) or ""

    def _clear_sel(self):
        if self._text_sel:
            self._text_sel = None
            self.viewport().update()

    def _copy_selected(self):
        if not self._text_sel:
            return
        _, start, end = self._text_sel
        text = self._row_text(self._text_sel[0])
        if text and start != end:
            QApplication.clipboard().setText(text[min(start, end):max(start, end)])
        self._clear_sel()

    def _copy_full_message(self, row=None):
        if row is None:
            return
        text = self._row_text(row)
        if text:
            QApplication.clipboard().setText(text)
        self._clear_sel()


# ============================================================
# 聊天面板
# ============================================================
class ChatPanel(QWidget):
    """聊天面板 — 消息列表 + 输入区域"""

    sig_send_message = pyqtSignal(str)  # 用户发送消息

    def __init__(self, session: AgentSession, async_run, parent=None):
        super().__init__(parent)
        self.session = session
        self.async_run = async_run

        self.model = ChatMessageModel()
        self._thinking_timer = QTimer(self)
        self._thinking_timer.timeout.connect(self._tick_thinking)
        self._thinking_step = 0
        self._thinking_states = ["○ ○ ○", "● ○ ○", "○ ● ○", "○ ○ ●"]
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # LLM 连接状态标签
        self.status_label = QLabel("⚪ 未连接 LLM — 请在设置中配置 API Key")
        self.status_label.setStyleSheet(
            "QLabel { color: #888; font-size: 11px; padding: 4px 10px; "
            "background: #0d0d1a; border-radius: 4px; }"
        )
        self.status_label.setFixedHeight(22)
        layout.addWidget(self.status_label)

        # 消息列表（支持文字选中）
        self.list_view = _SelectableListView()
        self.list_view.setModel(self.model)
        delegate = ChatBubbleDelegate(self.list_view)
        self.list_view.setItemDelegate(delegate)
        self.list_view.set_delegate(delegate)
        self.list_view.setSelectionMode(QListView.NoSelection)
        self.list_view.setVerticalScrollMode(QListView.ScrollPerPixel)
        self.list_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_view.setStyleSheet("QListView { background: #0a0a14; border: none; }")
        layout.addWidget(self.list_view, stretch=1)

        # 输入区域
        input_row = QHBoxLayout()
        input_row.setSpacing(6)

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("输入消息... (Enter 发送)")
        self.input_box.setStyleSheet("""
            QLineEdit {
                background: #1a1a2e;
                color: #ddd;
                border: 1px solid #333;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 13px;
            }
            QLineEdit:focus { border-color: #7c3aed; }
        """)
        self.input_box.returnPressed.connect(self._send)

        self.send_btn = QPushButton("发送")
        self.send_btn.setStyleSheet("""
            QPushButton {
                background: #7c3aed;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 18px;
                font-size: 13px;
            }
            QPushButton:hover { background: #6d28d9; }
            QPushButton:pressed { background: #5b21b6; }
        """)
        self.send_btn.clicked.connect(self._send)

        input_row.addWidget(self.input_box, stretch=1)
        input_row.addWidget(self.send_btn)
        layout.addLayout(input_row)

    def _send(self):
        """发送消息"""
        text = self.input_box.text().strip()
        if not text:
            return
        self.input_box.clear()

        # 显示用户消息
        self.model.add_message("user", text)
        self._scroll_to_bottom()

        # 调用 agent.chat()
        agent = self.session.agent
        if not agent:
            self.model.add_message("assistant", "⚠️ 请先连接 LLM")
            return

        # 如果有 pending autonomous，先追加
        pending = self.session.pop_pending_autonomous()
        for msg in pending:
            for bubble in msg.split("|||"):
                bubble = bubble.strip()
                if bubble:
                    self.model.add_message("assistant", bubble)

        # 动画占位符 — 三个暗淡圆点交叉亮起
        self._thinking_step = 0
        self.model.add_message("assistant", self._thinking_states[0])
        self._scroll_to_bottom()
        self._thinking_timer.start(400)

        def on_response(result):
            self._thinking_timer.stop()
            if isinstance(result, Exception):
                self.model.update_last(f"❌ 错误: {result}")
                return

            raw = result.text.strip() if hasattr(result, 'text') else str(result)
            # ReAct 模式：图片已在 agent 循环中生成完毕
            image_paths = getattr(result, 'image_paths', []) or []

            # 处理文字回复（可能含 ||| 多气泡）
            if "|||" in raw:
                bubbles = [b.strip() for b in raw.split("|||") if b.strip()]
                self.model.update_last(bubbles[0])
                for bubble in bubbles[1:]:
                    self.model.add_message("assistant", bubble)
            else:
                self.model.update_last(raw)
            self._scroll_to_bottom()

            # ---- ReAct 已生成图片：直接展示 ----
            for img_path in image_paths:
                if img_path:
                    self.model.add_message("assistant", "🖼", img_path)
                    self._scroll_to_bottom()

            # 自动保存 + 刷新情绪
            self.async_run(self.session.auto_save())
            self.async_run(self.session.refresh_display_state())

        self.async_run(agent.chat(text), on_response)

    def append_autonomous(self, message: str):
        """追加自主消息到聊天"""
        for bubble in message.split("|||"):
            bubble = bubble.strip()
            if bubble:
                self.model.add_message("assistant", bubble)
                if self.session.agent:
                    self.session.agent._add_to_history("assistant", bubble)
        self._scroll_to_bottom()

    def load_history(self, messages: list):
        """从存档加载对话历史"""
        for m in messages:
            self.model.add_message(m.role, m.content)
        self._scroll_to_bottom()

    def update_llm_status(self, connected: bool, provider: str = "", model: str = ""):
        """更新 LLM 连接状态标签"""
        img_status = ""
        if self.session._image_gen_provider:
            img_status = f"  |  🖼 {self.session._image_gen_provider}"
        if connected:
            info = f"{provider}"
            if model:
                info += f" ({model})"
            self.status_label.setText(f"🟢 已连接: {info}{img_status}")
            self.status_label.setStyleSheet(
                "QLabel { color: #8f8; font-size: 11px; padding: 4px 10px; "
                "background: #0d1a0d; border-radius: 4px; }"
            )
        else:
            base = "⚪ 未连接 LLM — 请在设置中配置 API Key"
            if img_status:
                base = "⚪ 未连接 LLM" + img_status + " — 请在设置中配置 API Key"
            self.status_label.setText(base)
            self.status_label.setStyleSheet(
                "QLabel { color: #888; font-size: 11px; padding: 4px 10px; "
                "background: #0d0d1a; border-radius: 4px; }"
            )

    def _tick_thinking(self):
        """动画：三个圆点循环亮起"""
        self._thinking_step = (self._thinking_step + 1) % len(self._thinking_states)
        self.model.update_last(self._thinking_states[self._thinking_step])

    def clear(self):
        """清空聊天"""
        self.model.clear()

    def _scroll_to_bottom(self):
        """滚动到底部"""
        QTimer.singleShot(50, lambda: self.list_view.scrollToBottom())
