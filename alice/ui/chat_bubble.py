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
    """渲染聊天气泡（文字 + 可选图片）+ 文字选中高亮 — Elysia 主题"""

    PADDING = 14
    BUBBLE_RADIUS = 16
    MAX_BUBBLE_WIDTH = 480
    THUMB_SIZE = 200
    _MAX_CACHE = 50

    # Elysia 主题色
    _USER_BUBBLE_BG = QColor("#c02669")      # 粉红 — 用户气泡
    _USER_BUBBLE_BG2 = QColor("#9d174d")     # 深粉 — 渐变辅助
    _AI_BUBBLE_BG = QColor("#13132b")        # 深紫黑 — AI 气泡
    _AI_BUBBLE_BORDER = QColor("#252545")    # AI 气泡边框
    _TEXT_USER = QColor("#fce7f3")           # 用户文字
    _TEXT_AI = QColor("#e2e8f0")             # AI 文字
    _TEXT_TIMESTAMP = QColor("#64748b")      # 时间戳
    _SELECTION_BG = QColor("#a855f7")        # 选中高亮

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
        margin_h = 16  # 水平边距

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
        doc_w = max(doc_w, 70.0)
        doc_h = self._doc.size().height() + self.PADDING * 2
        bubble_h = max(doc_h, 32.0)

        # ---- 气泡位置 ----
        if is_user:
            bx = full_w - doc_w - margin_h
        else:
            bx = margin_h + 28  # 给头像留空间

        by = 6
        total_h = bubble_h + img_h + (6 if img_h else 0)

        # ---- AI 头像指示 ----
        if not is_user:
            avatar_x = 10
            avatar_y = int(by + 8)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#a855f7"))
            painter.drawRoundedRect(QRect(avatar_x, avatar_y, 20, 20), 6, 6)
            painter.setPen(QColor("#fce7f3"))
            painter.setFont(QFont("Microsoft YaHei", 9))
            painter.drawText(QRect(avatar_x, avatar_y, 20, 20),
                           Qt.AlignCenter, "E")

        # ---- 气泡阴影（AI 气泡有边框） ----
        painter.setPen(Qt.NoPen)
        if is_user:
            # 用户气泡 — 纯粉紫填充
            painter.setBrush(self._USER_BUBBLE_BG)
        else:
            # AI 气泡 — 先画边框再画填充
            painter.setPen(self._AI_BUBBLE_BORDER)
            painter.setBrush(self._AI_BUBBLE_BG)
        bubble_rect = QRect(int(bx), int(by), int(doc_w), int(total_h))
        painter.drawRoundedRect(bubble_rect, self.BUBBLE_RADIUS, self.BUBBLE_RADIUS)

        # ---- 图片 ----
        if img_path and img_path in self._pixmap_cache:
            pm = self._pixmap_cache[img_path]
            img_x = int(bx + (doc_w - pm.width()) / 2)
            painter.drawPixmap(img_x, int(by + 8), pm)
            by += img_h + 6

        # ---- 文字（含选中高亮） ----
        text_x = int(bx + self.PADDING)
        text_y = int(by + self.PADDING)
        text_w = int(doc_w - self.PADDING * 2)

        # 缓存文字区域供 hit-test
        self._text_rects[index.row()] = QRect(text_x, text_y, text_w,
                                              int(self._doc.size().height()))

        self._doc.setTextWidth(text_w)

        sel = self._get_selection(index.row())
        text_color = self._TEXT_USER if is_user else self._TEXT_AI

        if sel:
            cursor = QTextCursor(self._doc)
            cursor.setPosition(sel[0])
            cursor.setPosition(sel[1], QTextCursor.KeepAnchor)

            sel_fmt = QTextCharFormat()
            sel_fmt.setBackground(self._SELECTION_BG)
            sel_fmt.setForeground(Qt.white)

            es = QAbstractTextDocumentLayout.Selection()
            es.cursor = cursor
            es.format = sel_fmt

            ctx = QAbstractTextDocumentLayout.PaintContext()
            ctx.selections = [es]
            ctx.palette.setColor(QPalette.Text, text_color)

            painter.save()
            painter.translate(text_x, text_y)
            self._doc.documentLayout().draw(painter, ctx)
            painter.restore()
        else:
            painter.save()
            painter.translate(text_x, text_y)
            self._doc.setDefaultFont(QFont("Microsoft YaHei", 10))
            painter.setPen(text_color)
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
        return QSize(w, int(max(doc_h + img_h + 20, 44)))
