"""
EmotionBar — 6轴情绪仪表条
竖式布局：每轴 emoji → 彩色条 → 数值，上下排列，互不重叠。
"""

from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPainter, QColor, QFont


def _value_color(val: float, vmin: float, vmax: float) -> QColor:
    """Elysia 水晶渐变: 蓝紫 → 粉紫 → 暖粉"""
    t = (val - vmin) / (vmax - vmin + 0.0001)
    t = max(0.0, min(1.0, t))
    # 冷色到暖色的过渡
    r = int(100 + 155 * t)
    g = int(80 + 100 * (1 - abs(t - 0.5) * 2))
    b = int(220 - 100 * t)
    return QColor(min(r, 255), min(g, 255), min(b, 255))


def _mood_emoji(v: float, a: float) -> str:
    if v >= 0.15 and a >= 0.15:   return "😊"
    elif v >= 0.15 and a < -0.15:  return "😌"
    elif v < -0.15 and a >= 0.15:  return "😤"
    elif v < -0.15 and a < -0.15:  return "😞"
    else:                          return "😐"


class EmotionBar(QWidget):
    """情绪仪表条 — Elysia 水晶主题"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(118)
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

        # ---- 背景 ----
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#0a0a18"))
        p.drawRoundedRect(QRectF(0, 0, w, h), 10, 10)

        # ---- 6 条仪表等宽排列 ----
        n = 6
        margin = 12
        spacing = 8
        total_spacing = margin * 2 + spacing * (n - 1)
        col_w = (w - total_spacing) / n

        gauge_h = h - 30
        bar_h = 14
        bar_y = 28

        gauges = [
            ("💛", "思念", state.get("connection", 0.0),      0.0, 1.0),
            ("🛡",  "骄傲", state.get("pride", 0.0),           -1.0, 1.0),
            ("💚", "愉悦", state.get("valence", 0.0),         -1.0, 1.0),
            ("⚡",  "唤醒", state.get("arousal", 0.0),         -1.0, 1.0),
            ("🎯", "沉浸", state.get("immersion", 0.0),        0.0, 1.0),
            ("🔥", "意愿", state.get("net_willingness", 0.0), 0.0, 1.0),
        ]

        for i, (emoji, name, val, vmin, vmax) in enumerate(gauges):
            cx = margin + i * (col_w + spacing)

            # emoji + 名称
            p.setPen(QColor("#94a3b8"))
            p.setFont(QFont("Microsoft YaHei", 9))
            p.drawText(QRectF(cx, 4, col_w, 20),
                       Qt.AlignCenter | Qt.AlignVCenter, f"{emoji} {name}")

            # 底色条
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#1a1a32"))
            p.drawRoundedRect(QRectF(cx, bar_y, col_w, bar_h), 7, 7)

            # 填充条
            t = (val - vmin) / (vmax - vmin + 0.0001)
            fill_w = max(6, int(col_w * max(0.0, min(1.0, t))))
            color = _value_color(val, vmin, vmax)
            p.setBrush(color)
            p.drawRoundedRect(QRectF(cx, bar_y, fill_w, bar_h), 7, 7)

            # 数值
            p.setPen(QColor("#cbd5e1"))
            p.setFont(QFont("Consolas", 9))
            p.drawText(QRectF(cx, bar_y + bar_h + 2, col_w, 16),
                       Qt.AlignCenter | Qt.AlignVCenter,
                       f"{val:+.2f}" if vmin < 0 else f"{val:.2f}")

        # ---- 下排：状态信息 ----
        bot_y = int(gauge_h) + 4
        bot_h = int(h - gauge_h - 4)

        # 左：LLM + 预设
        p.setPen(QColor("#64748b"))
        p.setFont(QFont("Microsoft YaHei", 8))
        llm_color = "#a855f7" if self._llm_connected else "#475569"
        llm_dot = "●" if self._llm_connected else "○"
        left_w = int(w * 0.32)
        p.setPen(QColor(llm_color))
        p.drawText(QRectF(margin, bot_y, 16, bot_h),
                   Qt.AlignLeft | Qt.AlignVCenter, llm_dot)
        p.setPen(QColor("#94a3b8"))
        p.drawText(QRectF(margin + 16, bot_y, left_w - 16, bot_h),
                   Qt.AlignLeft | Qt.AlignVCenter,
                   f"  {self._preset_name}")

        # 中：心情 + zone
        valence = state.get("valence", 0.0)
        arousal = state.get("arousal", 0.0)
        zone = diag.get("zone", "idle")
        zone_labels = {
            "idle": "空闲", "observation": "观察",
            "consider_contact": "想联系", "force_contact": "想联系",
        }
        zone_label = zone_labels.get(zone, zone)
        mid_x = margin + left_w
        mid_w = int(w * 0.36)
        p.setFont(QFont("Microsoft YaHei", 9))
        p.drawText(QRectF(mid_x, bot_y, mid_w, bot_h),
                   Qt.AlignCenter | Qt.AlignVCenter,
                   f"{_mood_emoji(valence, arousal)}  {zone_label}")

        # 右：时间
        vt = diag.get("virtual_time_min", 0.0)
        right_x = int(mid_x + mid_w)
        right_w = int(w - right_x - margin)
        p.setFont(QFont("Microsoft YaHei", 8))
        p.setPen(QColor("#64748b"))
        hours = int(vt // 60)
        mins = int(vt % 60)
        time_str = f"{hours}h {mins}m" if hours > 0 else f"{mins} min"
        p.drawText(QRectF(right_x, bot_y, right_w, bot_h),
                   Qt.AlignRight | Qt.AlignVCenter, f"⏱ {time_str}")

        p.end()
