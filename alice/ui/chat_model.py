"""
聊天数据模型 — ChatMessage + ChatMessageModel
从 chat_panel.py 提取
"""
from typing import List, Dict

from PyQt5.QtCore import Qt, QAbstractListModel, QModelIndex


class ChatMessage:
    """单条聊天消息"""
    __slots__ = ('role', 'content', 'image_path')

    def __init__(self, role: str, content: str, image_path: str = ""):
        self.role = role
        self.content = content
        self.image_path = image_path


class ChatMessageModel(QAbstractListModel):
    """聊天消息列表模型"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._messages: List[ChatMessage] = []

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        msg = self._messages[index.row()]
        if role == Qt.DisplayRole:
            return msg.content
        if role == Qt.UserRole:
            return msg.role
        if role == Qt.UserRole + 1:
            return msg.image_path
        return None

    def rowCount(self, parent=QModelIndex()):
        return len(self._messages)

    def add_message(self, role: str, content: str, image_path: str = ""):
        row = len(self._messages)
        self.beginInsertRows(QModelIndex(), row, row)
        self._messages.append(ChatMessage(role, content, image_path))
        self.endInsertRows()

    def update_last(self, content: str = "", image_path: str = None):
        """更新最后一条消息（用于流式显示 / 图片）"""
        if self._messages:
            if content:
                self._messages[-1].content = content
            if image_path is not None:
                self._messages[-1].image_path = image_path
            idx = self.index(len(self._messages) - 1)
            self.dataChanged.emit(idx, idx, [Qt.DisplayRole])

    def last_role(self) -> str:
        return self._messages[-1].role if self._messages else ""

    def clear(self):
        self.beginResetModel()
        self._messages.clear()
        self.endResetModel()

    def to_dict_list(self) -> List[Dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self._messages]
