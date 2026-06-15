"""
ReferencePanel — 人物风格参考图素材管理面板
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton,
    QLabel, QListWidget, QListWidgetItem, QFileDialog, QMessageBox, QMenu,
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QPixmap, QIcon

from alice.ui.reference import ReferenceManager


class ReferencePanel(QWidget):
    """素材管理面板 — 上传/查看/删除角色参考图"""

    sig_reference_changed = pyqtSignal(str)  # preset_name

    def __init__(self, session, async_run, parent=None):
        super().__init__(parent)
        self.session = session
        self.async_run = async_run
        self._mgr = ReferenceManager()
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # 标题
        title = QLabel("📸 人物素材库")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #eee;")
        layout.addWidget(title)

        desc = QLabel("上传角色参考图，图片生成时将基于素材保持人物风格一致性。")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(desc)

        # 人格选择
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("角色:"))
        self.preset_combo = QComboBox()
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        row1.addWidget(self.preset_combo, stretch=1)
        layout.addLayout(row1)

        # 上传按钮行 & 删除全部
        row2 = QHBoxLayout()
        self.upload_img_btn = QPushButton("📷 上传图片")
        self.upload_img_btn.clicked.connect(self._on_upload_images)
        row2.addWidget(self.upload_img_btn)

        self.upload_zip_btn = QPushButton("📦 上传 ZIP")
        self.upload_zip_btn.clicked.connect(self._on_upload_zip)
        row2.addWidget(self.upload_zip_btn)

        self.delete_all_btn = QPushButton("🗑 清空全部")
        self.delete_all_btn.clicked.connect(self._on_delete_all)
        row2.addWidget(self.delete_all_btn)
        layout.addLayout(row2)

        # 状态
        self.status_label = QLabel("请选择角色")
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.status_label)

        # 图片网格
        self.image_list = QListWidget()
        self.image_list.setViewMode(QListWidget.IconMode)
        self.image_list.setIconSize(QSize(140, 140))
        self.image_list.setResizeMode(QListWidget.Adjust)
        self.image_list.setSpacing(8)
        self.image_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_list.customContextMenuRequested.connect(self._on_context_menu)
        self.image_list.setStyleSheet(
            "QListWidget { background: #111; border: 1px solid #333; border-radius: 6px; }"
            "QListWidget::item { background: #1a1a2e; border-radius: 4px; margin: 2px; }"
            "QListWidget::item:hover { background: #2a2a4e; }"
        )
        layout.addWidget(self.image_list, stretch=1)

        # 按钮样式
        btn_style = (
            "QPushButton { background: #2a2a4e; color: #ddd; border: 1px solid #444;"
            " border-radius: 6px; padding: 8px 14px; font-size: 12px; }"
            "QPushButton:hover { background: #3a3a5e; }"
        )
        for btn in [self.upload_img_btn, self.upload_zip_btn, self.delete_all_btn]:
            btn.setStyleSheet(btn_style)

    # ---- 公开方法 ----

    def refresh_preset_list(self):
        """刷新人格下拉列表"""
        current = self.preset_combo.currentText()
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        for preset in self.session.preset_manager.list_all():
            self.preset_combo.addItem(preset)
        idx = self.preset_combo.findText(current)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)
        self.preset_combo.blockSignals(False)
        self._refresh_grid()

    # ---- 内部 ----

    def _current_preset(self) -> str:
        return self.preset_combo.currentText().strip()

    def _on_preset_changed(self, _name: str):
        self._refresh_grid()

    def _refresh_grid(self):
        self.image_list.clear()
        name = self._current_preset()
        if not name:
            self.status_label.setText("请选择角色")
            return
        images = self._mgr.get_images(name)
        for img in images:
            pm = QPixmap(str(img))
            if not pm.isNull():
                pm = pm.scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                item = QListWidgetItem(QIcon(pm), img.name)
                item.setToolTip(img.name)
                self.image_list.addItem(item)
        count = len(images)
        self.status_label.setText(f"📁 {name}: {count} 张素材图")
        self.sig_reference_changed.emit(name)

    def _on_upload_images(self):
        name = self._current_preset()
        if not name:
            QMessageBox.warning(self, "提示", "请先选择角色")
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择参考图",
            "", "图片 (*.jpg *.jpeg *.png *.webp *.bmp *.gif)"
        )
        if files:
            added = self._mgr.add_images(name, files)
            self.status_label.setText(f"✅ 已添加 {added} 张到 {name}")
            self._refresh_grid()

    def _on_upload_zip(self):
        name = self._current_preset()
        if not name:
            QMessageBox.warning(self, "提示", "请先选择角色")
            return
        zip_file, _ = QFileDialog.getOpenFileName(
            self, "选择 ZIP 压缩包", "", "ZIP 文件 (*.zip)"
        )
        if zip_file:
            try:
                added = self._mgr.extract_zip(name, zip_file)
                self.status_label.setText(f"✅ 已从 ZIP 提取 {added} 张到 {name}")
                self._refresh_grid()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"解压失败: {e}")

    def _on_context_menu(self, pos):
        item = self.image_list.itemAt(pos)
        if not item:
            return
        menu = QMenu(self)
        delete_action = menu.addAction("🗑 删除")
        action = menu.exec_(self.image_list.mapToGlobal(pos))
        if action == delete_action:
            name = self._current_preset()
            filename = item.text()
            self._mgr.delete_image(name, filename)
            self._refresh_grid()

    def _on_delete_all(self):
        name = self._current_preset()
        if not name:
            return
        reply = QMessageBox.warning(
            self, "确认", f"确定要删除 {name} 的全部素材图吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            count = self._mgr.delete_all(name)
            self.status_label.setText(f"🗑 已删除 {name} 的 {count} 张素材")
            self._refresh_grid()
