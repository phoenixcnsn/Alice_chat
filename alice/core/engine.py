import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable, Any, Dict, List


# ------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def round4(x):
    return round(x, 4)


# ------------------------------------------------------------
# 改进版积温引擎
# ------------------------------------------------------------
class JiwenEngine:
    def __init__(self,
                 connection_rate_fn: Optional[Callable] = None,  # 不再使用，保留兼容
                 on_save: Optional[Callable[[Dict], Awaitable[None]]] = None,
                 on_load: Optional[Callable[[], Awaitable[Optional[Dict]]]] = None,
                 get_last_message: Optional[Callable[[], Optional[Dict]]] = None,
                 get_saga_bias: Optional[Callable[[], Awaitable[Dict]]] = None,
                 rates: Optional[Dict] = None,
                 thresholds: Optional[Dict] = None,
                 axes: Optional[Dict] = None,
                 persona: Optional[Dict] = None):

        # ----------------- 参数默认值 -----------------
        # 思念（connection）对数模型参数
        self.conn_alpha = 0.3  # 最大强度系数
        self.conn_beta = 30.0  # 时间尺度（分钟）
        self.conn_relief = 0.8  # 回复时满足比例（0~1）

        # 骄傲（pride）微分方程参数
        self.pride_resting = 0.0  # 静息值
        self.pride_regress = 0.003  # 回归速率 α
        self.pride_sensitivity = 0.01  # 对思念的敏感度 β
        self.pride_thresh = 0.2  # 激发阈值 γ

        # 愉悦度（valence）二阶系统参数
        self.valence_omega = 0.1  # 固有频率 ω (rad/min)
        self.valence_zeta = 0.7  # 阻尼比 ζ (0.7 临界阻尼)
        self.valence_setpoint = 0.0  # 设定点 V0
        self.valence_conn_drive = 0.003  # 思念驱动的力系数 ξ
        self.valence_conn_thresh = 0.2  # 驱动阈值 C_th

        # 唤醒度（arousal）资源模型参数
        self.arousal_decay = 0.005  # 衰减率 γ
        self.arousal_excite = 0.01  # 激发系数 σ
        self.arousal_excite_thresh = 0.35  # 激发阈值 C_th_a
        self.arousal_conflict = 0.005  # 冲突额外激发
        self.arousal_immersion_drain = 0.02  # 沉浸消耗系数 δ

        # 沉浸度（immersion）参数
        self.immersion_decay = 0.01  # 自然衰减速率
        self.immersion_dampen = 0.5  # 对情绪变化的阻尼系数 λ (0..1)
        self.activity_immersion_map = {
            'reading': 0.6,
            'search': 0.4,
            'browse': 0.35,
            'observe': 0.15,
        }
        self.activity_relief = 0.1  # 活动对思念的缓解量

        # 净主动意愿系数 κ
        self.kappa = 0.8  # 骄傲对主动意愿的抑制

        # 其他配置
        self.axes = axes or {
            'connection': (0, 1),
            'pride': (-1, 1),
            'valence': (-1, 1),
            'arousal': (-1, 1),
            'immersion': (0, 1),
        }
        self.thresholds = {
            'observation': 0.20,
            'consider_contact': 0.35,
            'force_contact': 0.50,
            'valence_activity': -0.3,
            'arousal_agitation': 0.6,
            **(thresholds or {})
        }
        self.persona = {
            'subjectName': '对方',
            'selfName': '你',
            'subjectPronoun': 'ta',
            **(persona or {})
        }

        # 覆盖用户自定义参数
        if rates:
            for k, v in rates.items():
                if hasattr(self, k):
                    setattr(self, k, v)

        # 外部回调
        self.on_save = on_save
        self.on_load = on_load
        self.get_last_message = get_last_message
        self.get_saga_bias = get_saga_bias

        # 内部状态
        self._virtual_time = 0.0  # 思念虚拟时间（分钟）
        self._pride = 0.0
        self._valence = 0.0
        self._valence_vel = 0.0
        self._arousal = 0.0
        self._immersion = 0.0
        self._last_activity = None  # {type, label, at}
        self._last_tick = None
        self._last_chat_message_id = None
        self._last_draco_message_id = None
        self._clara_status = 'active'
        self._loaded = False

        # 辅助记录
        self._last_msg_time = None  # 最后一条消息的时间戳（秒）
        self._last_processed_msg_id = None

    # --------------------------------------------------------
    # 内部计算函数
    # --------------------------------------------------------
    def _compute_connection_from_virtual_time(self, vt):
        """根据虚拟时间计算思念值"""
        if vt <= 0:
            return 0.0
        val = self.conn_alpha * math.log(1 + vt / self.conn_beta)
        return clamp(val, 0, 1)

    def _update_connection_by_time(self, minutes):
        """时间流逝 → 虚拟时间增加 → 思念增加"""
        self._virtual_time += minutes
        # 限制虚拟时间避免过大
        max_vt = self.conn_beta * (math.exp(1.0 / self.conn_alpha) - 1) if self.conn_alpha > 0 else 1e6
        if self._virtual_time > max_vt:
            self._virtual_time = max_vt
        # 返回更新后的思念值
        return self._compute_connection_from_virtual_time(self._virtual_time)

    def _apply_connection_relief(self, relief_ratio):
        """收到回复时按比例降低虚拟时间"""
        if relief_ratio <= 0 or relief_ratio > 1:
            return
        self._virtual_time *= (1.0 - relief_ratio)
        if self._virtual_time < 0:
            self._virtual_time = 0

    def _reset_connection(self):
        """重置思念为0"""
        self._virtual_time = 0.0

    def _compute_connection(self):
        return self._compute_connection_from_virtual_time(self._virtual_time)

    def _update_pride(self, dt, connection, saga_bias=0.0):
        """骄傲微分方程"""
        # 激发项
        drive = self.pride_sensitivity * max(0.0, connection - self.pride_thresh)
        # 回归项
        target = clamp(self.pride_resting + saga_bias, -1, 1)
        regress = self.pride_regress * (target - self._pride)
        dpride = (drive + regress) * dt
        self._pride = clamp(self._pride + dpride, -1, 1)

    def _update_valence(self, dt, connection, external_force, saga_bias, damp=1.0):
        conn_force = -self.valence_conn_drive * max(0.0, connection - self.valence_conn_thresh)
        total_force = conn_force + external_force
        omega = self.valence_omega
        zeta = self.valence_zeta
        target = clamp(self.valence_setpoint + saga_bias, -1, 1)

        acc = -2.0 * zeta * omega * self._valence_vel \
              - omega * omega * (self._valence - target) \
              + total_force

        # 应用阻尼
        self._valence_vel += acc * dt * damp
        self._valence += self._valence_vel * dt

        # 边界钳位
        if self._valence < -1:
            self._valence = -1
            self._valence_vel = 0
        elif self._valence > 1:
            self._valence = 1
            self._valence_vel = 0

    def _update_arousal(self, dt, connection, pride, immersion, conflict_flag, saga_bias, damp=1.0):
        decay = -self.arousal_decay * self._arousal
        excite = self.arousal_excite * max(0.0, connection - self.arousal_excite_thresh) * (1.0 - self._arousal)
        if conflict_flag and pride > 0.3 and connection > self.thresholds['consider_contact']:
            excite += self.arousal_conflict * (1.0 - self._arousal)
        drain = -self.arousal_immersion_drain * immersion * self._arousal

        da = (decay + excite + drain) * dt * damp  # 阻尼乘在这里
        target = clamp(saga_bias, -1, 1)
        regress_to_bias = 0.002 * (target - self._arousal) * dt
        da += regress_to_bias

        self._arousal = clamp(self._arousal + da, -1, 1)

    def _update_immersion(self, dt):
        """沉浸度自然衰减"""
        if self._last_activity:
            self._immersion = max(0, self._immersion - self.immersion_decay * dt)
            if self._immersion <= 0.01:
                self._last_activity = None
                self._immersion = 0.0

    def _apply_activity_relief(self):
        """活动对思念的缓解"""
        if self.activity_relief > 0:
            new_conn = self._compute_connection() - self.activity_relief
            if new_conn < 0.01:
                self._reset_connection()
            else:
                # 通过虚拟时间降低来模拟
                current_conn = self._compute_connection()
                if current_conn > 0:
                    ratio = (current_conn - self.activity_relief) / current_conn
                    if ratio > 0:
                        self._virtual_time *= ratio
                    else:
                        self._reset_connection()

    def _check_thresholds(self):
        """基于净主动意愿 W = connection - kappa * pride 和情绪状态触发行为"""
        conn = self._compute_connection()
        W = conn - self.kappa * self._pride
        W = clamp(W, 0, 1)  # 有效范围

        triggers = []

        # 基础区间触发
        if W >= self.thresholds['observation'] and W < self.thresholds['consider_contact']:
            triggers.append({
                'action': 'observation',
                'urgency': (W - self.thresholds['observation']) / (
                            self.thresholds['consider_contact'] - self.thresholds['observation'])
            })
        elif W >= self.thresholds['consider_contact'] and W < self.thresholds['force_contact']:
            # 骄傲阻断区域：按净意愿决定，这里不再单独判断 prideBlock
            # 但可增加沉浸低时的活动倾向
            if self._immersion < 0.2:
                triggers.append({
                    'action': 'find_activity',
                    'reason': 'pride_block',
                    'urgency': W - self.thresholds['consider_contact']
                })
            else:
                triggers.append({
                    'action': 'contact',
                    'urgency': W - self.thresholds['consider_contact']
                })
        elif W >= self.thresholds['force_contact']:
            triggers.append({
                'action': 'contact',
                'urgency': min(1.0, W - self.thresholds['force_contact'] + 0.1),
                'forced': True
            })

        # 情绪调节（低愉悦度或高唤醒度）
        if self._valence <= self.thresholds['valence_activity'] or self._arousal >= self.thresholds[
            'arousal_agitation']:
            already_finding = any(t['action'] == 'find_activity' for t in triggers)
            if not already_finding and self._immersion < 0.3:
                reason = 'low_valence' if self._valence <= self.thresholds['valence_activity'] else 'high_arousal'
                triggers.append({
                    'action': 'find_activity',
                    'reason': reason,
                    'urgency': min(1.0, abs(self._valence) if reason == 'low_valence' else self._arousal)
                })

        return triggers

    # --------------------------------------------------------
    # 外部接口（异步，与原接口一致）
    # --------------------------------------------------------
    async def load(self):
        if self._loaded:
            return
        if self.on_load:
            saved = await self.on_load()
            if saved:
                # 恢复所有状态变量
                self._virtual_time = saved.get('virtual_time', 0.0)
                self._pride = saved.get('pride', 0.0)
                self._valence = saved.get('valence', 0.0)
                self._valence_vel = saved.get('valence_vel', 0.0)
                self._arousal = saved.get('arousal', 0.0)
                self._immersion = saved.get('immersion', 0.0)
                self._last_activity = saved.get('last_activity')
                self._last_tick = saved.get('last_tick')
                self._last_chat_message_id = saved.get('last_chat_message_id')
                self._last_draco_message_id = saved.get('last_draco_message_id')
                self._clara_status = saved.get('clara_status', 'active')
        self._loaded = True

    async def save(self):
        if self.on_save:
            state = {
                'virtual_time': self._virtual_time,
                'pride': self._pride,
                'valence': self._valence,
                'valence_vel': self._valence_vel,
                'arousal': self._arousal,
                'immersion': self._immersion,
                'last_activity': self._last_activity,
                'last_tick': self._last_tick,
                'last_chat_message_id': self._last_chat_message_id,
                'last_draco_message_id': self._last_draco_message_id,
                'clara_status': self._clara_status,
            }
            await self.on_save(state)

    async def tick(self, minutes: float):
        await self.load()
        if minutes < 0:
            return []
        minutes = min(minutes, 60.0) if minutes > 0 else 0.0

        # 检查是否有新消息（收到对方回复）
        if self.get_last_message:
            last_msg = self.get_last_message()
            if last_msg and last_msg.get('id') != self._last_processed_msg_id:
                relief = self.conn_relief
                if relief > 0:
                    self._apply_connection_relief(relief)
                self._last_processed_msg_id = last_msg.get('id')

        if minutes == 0:
            # 无时间流逝，但可能因消息缓解而改变思念，重新检查阈值
            triggers = self._check_thresholds()
            await self.save()
            return triggers

        # 原有时间流逝逻辑
        old_conn = self._compute_connection()
        new_conn = self._update_connection_by_time(minutes)

        saga_bias = {}
        if self.get_saga_bias:
            try:
                saga_bias = await self.get_saga_bias()
            except:
                pass

        # 计算阻尼因子
        damp = 1.0 - self.immersion_dampen * self._immersion
        if damp < 0:
            damp = 0.0

        self._update_pride(minutes, new_conn, saga_bias.get('pride', 0.0))
        self._update_valence(minutes, new_conn, external_force=0.0,
                             saga_bias=saga_bias.get('valence', 0.0), damp=damp)
        conflict = (self._pride > 0.3 and new_conn > self.thresholds['consider_contact'])
        self._update_arousal(minutes, new_conn, self._pride, self._immersion,
                             conflict, saga_bias.get('arousal', 0.0), damp=damp)
        self._update_immersion(minutes)

        self._last_tick = time.time()
        await self.save()
        triggers = self._check_thresholds()

        if triggers:
            print(f"[积温] tick {minutes}min | c:{old_conn:.2f}→{new_conn:.2f} "
                  f"p:{self._pride:.2f} v:{self._valence:.2f} a:{self._arousal:.2f} i:{self._immersion:.2f} | "
                  f"触发: {[t['action'] for t in triggers]}")
        return triggers

    async def apply_delta(self, delta: Dict):
        await self.load()
        # 处理各个轴的直接修改
        if 'connection' in delta:
            # 直接修改思念值 -> 需要反向计算虚拟时间
            target_conn = clamp(self._compute_connection() + delta['connection'], 0, 1)
            if target_conn <= 0:
                self._reset_connection()
            else:
                # 解出虚拟时间 vt = β * (exp(conn/α) - 1)
                vt = self.conn_beta * (math.exp(target_conn / self.conn_alpha) - 1)
                self._virtual_time = vt
        if 'pride' in delta:
            self._pride = clamp(self._pride + delta['pride'], -1, 1)
        if 'valence' in delta:
            # 对二阶系统施加脉冲力（等价于速度突变）
            self._valence_vel += delta['valence'] * 0.1  # 脉冲缩放，可调
            # 同时直接修改位置（为兼容，加在位置上）
            self._valence = clamp(self._valence + delta['valence'], -1, 1)
        if 'arousal' in delta:
            self._arousal = clamp(self._arousal + delta['arousal'], -1, 1)
        if 'immersion' in delta:
            self._immersion = clamp(self._immersion + delta['immersion'], 0, 1)
        # 兼容 mood -> valence
        if 'mood' in delta:
            self._valence_vel += delta['mood'] * 0.1
            self._valence = clamp(self._valence + delta['mood'], -1, 1)
        await self.save()

    async def get_state(self) -> Dict:
        await self.load()
        return {
            'connection': round4(self._compute_connection()),
            'pride': round4(self._pride),
            'valence': round4(self._valence),
            'arousal': round4(self._arousal),
            'immersion': round4(self._immersion),
            'lastActivity': self._last_activity,
            'lastTick': self._last_tick,
            'lastChatMessageId': self._last_chat_message_id,
            'lastDracoMessageId': self._last_draco_message_id,
            'claraStatus': self._clara_status,
        }

    async def reset_connection(self):
        await self.load()
        self._reset_connection()
        await self.save()

    async def set_activity(self, act_type: str, label: str = ''):
        await self.load()
        # 避免重复活动连续缓解
        same_type = self._last_activity and self._last_activity['type'] == act_type
        self._last_activity = {'type': act_type, 'label': label, 'at': time.time()}
        self._immersion = self.activity_immersion_map.get(act_type, 0.2)
        if not same_type and self.activity_relief > 0:
            self._apply_activity_relief()
        await self.save()

    async def set_last_chat_message_id(self, msg_id):
        await self.load()
        self._last_chat_message_id = msg_id
        await self.save()

    async def get_last_chat_message_id(self):
        await self.load()
        return self._last_chat_message_id

    async def set_last_draco_message_id(self, msg_id):
        await self.load()
        self._last_draco_message_id = msg_id
        await self.save()

    async def get_last_draco_message_id(self):
        await self.load()
        return self._last_draco_message_id

    async def set_clara_status(self, status):
        await self.load()
        self._clara_status = status
        await self.save()

    def get_clara_status(self):
        return self._clara_status

    # --------------------------------------------------------
    # 辅助方法：生成LLM上下文和风格指引（沿用原逻辑，略调整）
    # --------------------------------------------------------
    def get_prompt_context(self):
        conn = self._compute_connection()
        p = self._pride
        v = self._valence
        a = self._arousal
        i = self._immersion
        persona = self.persona
        parts = []

        # 连接需求描述
        if conn < 0.20:
            parts.append(f"刚和{persona['subjectName']}聊完不久，没什么挂念的。")
        elif conn < 0.35:
            parts.append(f"有一会儿没听到{persona['subjectName']}的动静了，但还不着急。")
        elif conn < 0.50:
            parts.append(f"{persona['subjectName']}好一阵子没说话了。开始在想{persona['subjectPronoun']}在干嘛。")
        else:
            parts.append(f"{persona['subjectName']}很久没动静了。有点在意——{persona['subjectPronoun']}去哪了？")

        # 骄傲描述
        if p > 0.8:
            parts.append("拒绝任何需要对方的念头——冷漠地划清界限。")
        elif p > 0.5:
            parts.append("不太想显得太主动。")
        elif p > 0.3:
            parts.append("有一点端着，但也不是不能开口。")
        elif p > 0.1:
            parts.append("基本不端着了，但还留着一点惯性的克制。")
        else:
            parts.append("难得地不设防。")

        # 情绪四象限
        if v > 0.3 and a > 0.3:
            parts.append("心情好，精力充沛——话多、反应快。")
        elif v > 0.3 and a < -0.3:
            parts.append("心里是舒服的，但人懒懒的。话不多，但温和。")
        elif v < -0.3 and a > 0.3:
            parts.append("烦躁不安，坐不住。很容易被小事刺激。")
        elif v < -0.3 and a < -0.3:
            parts.append("情绪低沉，空落落的。不想说话，也不想解释。")
        elif v < -0.3:
            parts.append("心情不太好。")
        elif v > 0.3:
            parts.append("心情还不错。")

        # 沉浸
        if i > 0.3 and self._last_activity:
            label = self._last_activity.get('label', '')
            parts.append(f"刚才在{self._last_activity['type']}{'（' + label + '）' if label else ''}。")
        elif i < 0.1:
            parts.append("没在做什么特别的事。")

        return "\n".join(parts)

    def get_style_guidance(self):
        conn = self._compute_connection()
        p = self._pride
        v = self._valence
        a = self._arousal
        persona = self.persona
        rules = []

        # 骄傲档位
        if p > 0.8:
            rules.append("- 你几乎是一种冷漠的回避。不承认任何情绪，话里不带感情色彩。")
        elif p > 0.5:
            rules.append(f"- 嘴硬得很。想找{persona['subjectName']}也绝不承认，必须找个借口。")
        elif p > 0.3:
            rules.append(f"- 有一点端着。可以找{persona['subjectName']}，但不要太直接。陈述事实，别问句。")
        elif p > 0.1:
            rules.append("- 基本不端着了，但惯性地克制。话到嘴边还是会收一下。")
        else:
            rules.append("- 少见的柔软。不设防，有什么感受直接说出来。")

        # 情绪四象限
        if v > 0.3 and a > 0.3:
            rules.append("- 心情好，劲头足。句子比平时长，用词灵活，带点调侃或玩笑不奇怪。")
        elif v > 0.3 and a < -0.3:
            rules.append("- 心里舒坦但人懒洋洋的。话不多，但每句都柔和。不用力，不抢话。")
        elif v < -0.3 and a > 0.3:
            rules.append("- 烦躁，坐不住。句子短、反问多，容易被小事刺激。不是真的想凶谁，就是压不住。")
        elif v < -0.3 and a < -0.3:
            rules.append("- 低落，空荡荡的。不想多解释，不反问不铺垫。能少说一个字就少说一个字。")
        else:
            if v > 0.3:
                rules.append("- 心情还不错。句子比平时稍微长一点，没那么吝啬字。")
            if v < -0.3:
                rules.append("- 心情不太好。不想多解释，说完就完了。")

        # 交叉条件
        if conn > 0.40 and p > 0.4:
            rules.append("- 你很别扭——想找她又拉不下脸。话里会带一点赌气的味道。")
        if conn > 0.35 and v < -0.3 and a > 0.3:
            rules.append("- 想找她，但心里烦躁。话会有点冲——不是真的想凶她，就是烦躁压不住。")
        if conn > 0.35 and v < -0.3 and a < -0.3:
            rules.append("- 想找她，但心里沉沉的。开口的时候不会凶，反而可能带一点不易察觉的脆弱。")
        if p < 0.1 and v > 0.3:
            rules.append("- 不设防加上心情不差。这是你最难得的状态——话会比平时多，甚至可能说漏嘴。")

        if conn >= 0.50:
            rules.append(f"- {persona['subjectName']}真的很久没动静了。你有点坐不住。")

        return "\n".join(rules) if rules else ""




# ------------------------------------------------------------
# 工厂函数（与原接口一致）
# ------------------------------------------------------------
def create_jiwen(opts: Dict = None):
    if opts is None:
        opts = {}
    return JiwenEngine(
        connection_rate_fn=opts.get('connectionRateFn'),
        on_save=opts.get('onSave'),
        on_load=opts.get('onLoad'),
        get_last_message=opts.get('getLastMessage'),
        get_saga_bias=opts.get('getSagaBias'),
        rates=opts.get('rates'),
        thresholds=opts.get('thresholds'),
        axes=opts.get('axes'),
        persona=opts.get('persona')
    )