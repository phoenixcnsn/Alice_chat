"""
EmotionBar — 6轴情绪仪表条
竖式布局：每轴 emoji → 彩色条 → 数值，上下排列，互不重叠。
"""

from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPainter, QColor, QFont


def _value_color(val: float, vmin: float, vmax: float) -> QColor:
    t = (val - vmin) / (vmax - vmin + 0.0001)
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return QColor(int(255 * t * 2), 200, 60)
    else:
        return QColor(255, int(200 * (1 - (t - 0.5) * 2)), 30)


def _mood_emoji(v: float, a: float) -> str:
    if v >= 0 and a >= 0:   return "😊"
    elif v >= 0 and a < 0:  return "😌"
    elif v < 0 and a >= 0:  return "😤"
    else:                   return "😞"


class EmotionBar(QWidget):
    """情绪仪表条 — 每轴独立竖列，互不重叠"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(110)
        self.setMinimumWidth(500)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._state = {}
        self._diag = {}
        self._llm_connected = False
        self._preset_name = "默认"
        self._last_activity = None

    def update_state(self, state: dict, diag: dict, llm_connected: bool,
                     preset_name: str, last_activity):
        self._state = state or {}
        self._diag = diag or {}
        self._llm_connected = llm_connected
        self._preset_name = preset_name
        self._last_activity = last_activity
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        state = self._state
        diag = self._diag

        # ---- 6 条仪表等宽排列 ----
        n = 6
        margin = 10
        spacing = 10
        total_spacing = margin * 2 + spacing * (n - 1)
        col_w = (w - total_spacing) / n

        # 仪表区域高度（留底部状态行）
        gauge_h = h - 28
        bar_h = 16
        bar_y = 26  # emoji 下方

        gauges = [
            ("💛", "思念", state.get("connection", 0.0),      0.0, 1.0),
            ("🛡", "骄傲", state.get("pride", 0.0),           -1.0, 1.0),
            ("💚", "愉悦", state.get("valence", 0.0),         -1.0, 1.0),
            ("⚡", "唤醒", state.get("arousal", 0.0),         -1.0, 1.0),
            ("🎯", "沉浸", state.get("immersion", 0.0),        0.0, 1.0),
            ("🔥", "意愿", state.get("net_willingness", 0.0), 0.0, 1.0),
        ]

        # 上排：表盘
        for i, (emoji, name, val, vmin, vmax) in enumerate(gauges):
            cx = margin + i * (col_w + spacing)
            cy = 4

            # emoji + 名称
            p.setPen(QColor("#ccc"))
            p.setFont(QFont("Microsoft YaHei", 10))
            p.drawText(QRectF(cx, cy, col_w, 20),
                       Qt.AlignCenter | Qt.AlignVCenter, f"{emoji}{name}")

            # 彩色条
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#2a2a3a"))
            p.drawRoundedRect(QRectF(cx, bar_y, col_w, bar_h), 5, 5)

            t = (val - vmin) / (vmax - vmin + 0.0001)
            fill_w = max(4, int(col_w * max(0.0, min(1.0, t))))
            p.setBrush(_value_color(val, vmin, vmax))
            p.drawRoundedRect(QRectF(cx, bar_y, fill_w, bar_h), 5, 5)

            # 数值
            p.setPen(QColor("#ddd"))
            p.setFont(QFont("Consolas", 10))
            p.drawText(QRectF(cx, bar_y + bar_h + 2, col_w, 18),
                       Qt.AlignCenter | Qt.AlignVCenter,
                       f"{val:+.2f}" if vmin < 0 else f"{val:.2f}")

        # ---- 下排：状态信息 ----
        bot_y = int(gauge_h) + 2
        bot_h = int(h - gauge_h - 4)

        # 左：LLM + 预设
        p.setPen(QColor("#888"))
        p.setFont(QFont("Microsoft YaHei", 9))
        llm_dot = "🟢" if self._llm_connected else "⚪"
        left_w = int(w * 0.35)
        p.drawText(QRectF(margin, bot_y, left_w, bot_h),
                   Qt.AlignLeft | Qt.AlignVCenter,
                   f"{llm_dot}  {self._preset_name}")

        # 中：心情 + zone
        valence = state.get("valence", 0.0)
        arousal = state.get("arousal", 0.0)
        zone = diag.get("zone", "idle")
        mid_x = int(w * 0.30)
        mid_w = int(w * 0.40)
        p.setFont(QFont("Microsoft YaHei", 10))
        p.drawText(QRectF(mid_x, bot_y, mid_w, bot_h),
                   Qt.AlignCenter | Qt.AlignVCenter,
                   f"{_mood_emoji(valence, arousal)}  {zone}")

        # 右：时间
        vt = diag.get("virtual_time_min", 0.0)
        right_x = int(mid_x + mid_w)
        right_w = int(w - right_x - margin)
        p.setFont(QFont("Microsoft YaHei", 9))
        p.drawText(QRectF(right_x, bot_y, right_w, bot_h),
                   Qt.AlignRight | Qt.AlignVCenter,
                   f"⏱ {vt:.0f} min")

        p.end()
