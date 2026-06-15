"""
聊天气泡委托 — ChatBubbleDelegate
支持文字鼠标选中、右键复制
"""
from PyQt5.QtWidgets import (
    QStyledItemDelegate, QStyleOptionViewItem, QMenu, QApplication,
)
from PyQt5.QtCore import Qt, QModelIndex, QSize, QRect
from PyQt5.QtGui import (
    QPainter, QColor, QFont, QTextDocument, QTextCursor,
    QAbstractTextDocumentLayout, QPalette, QTextCharFormat,
)


class ChatBubbleDelegate(QStyledItemDelegate):
    """渲染聊天气泡（文字 + 可选图片）+ 文字选中高亮"""

    PADDING = 8
    BUBBLE_RADIUS = 12
    MAX_BUBBLE_WIDTH = 500
    THUMB_SIZE = 200
    _MAX_CACHE = 50

    def __init__(self, parent=None):
        super().__init__(parent)
        self._doc = QTextDocument()
        self._pixmap_cache: dict = {}
        # 缓存每行的文字布局坐标，供 view 做 hit-test
        self._text_rects: dict = {}   # row -> QRect (在 item 本地坐标系中)

    # ---- 供外部查询 ----
    def text_rect_for_row(self, row: int) -> QRect:
        """返回 row 的文字区域（item 本地坐标），无缓存返回空"""
        return self._text_rects.get(row, QRect())

    def document(self) -> QTextDocument:
        return self._doc

    # ---- 选择状态 ----
    def _get_selection(self, row: int):
        """从父 view 获取当前文字选择范围"""
        v = self.parent()
        if hasattr(v, '_text_sel') and v._text_sel:
            sel_row, start, end = v._text_sel
            if sel_row == row and start != end:
                return min(start, end), max(start, end)
        return None

    # ================================================================
    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)

        role = index.data(Qt.UserRole)
        text = index.data(Qt.DisplayRole) or ""
        img_path = index.data(Qt.UserRole + 1) or ""
        is_user = (role == "user")

        painter.translate(option.rect.topLeft())
        full_w = option.rect.width()

        # ---- 图片缩略图 ----
        img_h = 0
        if img_path and img_path in self._pixmap_cache:
            img_h = self.THUMB_SIZE
        elif img_path:
            from PyQt5.QtGui import QPixmap
            pm = QPixmap(img_path)
            if not pm.isNull():
                pm = pm.scaledToWidth(min(self.THUMB_SIZE, full_w * 0.5),
                                      Qt.SmoothTransformation)
                self._pixmap_cache[img_path] = pm
                if len(self._pixmap_cache) > self._MAX_CACHE:
                    self._pixmap_cache.pop(next(iter(self._pixmap_cache)))
                img_h = pm.height()

        # ---- 布局文字 ----
        self._doc.setDefaultFont(QFont("Microsoft YaHei", 10))
        self._doc.setPlainText(text)
        self._doc.setTextWidth(self.MAX_BUBBLE_WIDTH)
        doc_w = min(self._doc.idealWidth() + self.PADDING * 2,
                    self.MAX_BUBBLE_WIDTH + self.PADDING * 2)
        if img_h:
            doc_w = max(doc_w, float(self.THUMB_SIZE))
        doc_w = max(doc_w, 60.0)
        doc_h = self._doc.size().height() + self.PADDING * 2
        bubble_h = max(doc_h, 28.0)

        # ---- 气泡位置 ----
        bx = full_w - doc_w - 12 if is_user else 12
        by = 4
        total_h = bubble_h + img_h + (4 if img_h else 0)

        # ---- 气泡背景 ----
        painter.setPen(Qt.NoPen)
        bg_color = QColor("#3a1a5c") if is_user else QColor("#1a1a2e")
        painter.setBrush(bg_color)
        bubble_rect = QRect(int(bx), int(by), int(doc_w), int(total_h))
        painter.drawRoundedRect(bubble_rect, self.BUBBLE_RADIUS, self.BUBBLE_RADIUS)

        # ---- 图片 ----
        if img_path and img_path in self._pixmap_cache:
            pm = self._pixmap_cache[img_path]
            img_x = int(bx + (doc_w - pm.width()) / 2)
            painter.drawPixmap(img_x, int(by + 6), pm)
            by += img_h + 4

        # ---- 文字（含选中高亮） ----
        text_x = int(bx + self.PADDING)
        text_y = int(by + self.PADDING)
        text_w = int(doc_w - self.PADDING * 2)

        # 缓存文字区域供 hit-test
        self._text_rects[index.row()] = QRect(text_x, text_y, text_w,
                                              int(self._doc.size().height()))

        self._doc.setTextWidth(text_w)

        sel = self._get_selection(index.row())
        painter.setPen(Qt.white if is_user else QColor("#ddd"))

        if sel:
            # ---- 有选中：用 PaintContext 渲染高亮 ----
            cursor = QTextCursor(self._doc)
            cursor.setPosition(sel[0])
            cursor.setPosition(sel[1], QTextCursor.KeepAnchor)

            sel_fmt = QTextCharFormat()
            sel_fmt.setBackground(QColor("#7c3aed"))
            sel_fmt.setForeground(Qt.white)

            es = QAbstractTextDocumentLayout.Selection()
            es.cursor = cursor
            es.format = sel_fmt

            ctx = QAbstractTextDocumentLayout.PaintContext()
            ctx.selections = [es]
            ctx.palette.setColor(QPalette.Text, Qt.white if is_user else QColor("#ddd"))

            painter.save()
            painter.translate(text_x, text_y)
            self._doc.documentLayout().draw(painter, ctx)
            painter.restore()
        else:
            painter.save()
            painter.translate(text_x, text_y)
            self._doc.drawContents(painter)
            painter.restore()

        painter.restore()

    # ================================================================
    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        text = index.data(Qt.DisplayRole) or ""
        img_path = index.data(Qt.UserRole + 1) or ""
        w = option.widget.width() if option.widget else 400

        img_h = self.THUMB_SIZE if img_path else 0

        self._doc.setDefaultFont(QFont("Microsoft YaHei", 10))
        self._doc.setPlainText(text)
        self._doc.setTextWidth(min(self.MAX_BUBBLE_WIDTH, w * 0.7))
        doc_h = self._doc.size().height() + self.PADDING * 2
        return QSize(w, int(max(doc_h + img_h + 16, 36)))
