"""
角色特征提取器 —— 通过 LLM 分析文本提取角色人格画像

将对话、角色描述、故事片段等文本送入 LLM，
自动分析并输出结构化的 CharacterProfile，
包含积温引擎参数映射、说话风格、行为模式等。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from alice.core.agent import LLMCall


# ------------------------------------------------------------
# 引擎全部可调参数（用于 LLM 映射参考）
# ------------------------------------------------------------
ENGINE_PARAMS_SPEC: Dict[str, Dict[str, Any]] = {
    # 思念 (connection)
    "conn_alpha":       {"default": 0.3,   "range": (0.05, 0.6),  "desc": "最大思念强度系数，越大思念增长越快"},
    "conn_beta":        {"default": 30.0,  "range": (10.0, 120.0),"desc": "思念时间尺度(分钟)，越大越不容易思念"},
    "conn_relief":      {"default": 0.8,   "range": (0.3, 1.0),   "desc": "收到回复时思念缓解比例，越大缓解越多"},
    # 骄傲 (pride)
    "pride_resting":    {"default": 0.0,   "range": (-0.5, 0.5),  "desc": "骄傲静息值，正值=天生端着，负值=天生不设防"},
    "pride_regress":    {"default": 0.003, "range": (0.001, 0.01),"desc": "骄傲回归速率，越大越快回到静息值"},
    "pride_sensitivity":{"default": 0.01,  "range": (0.002, 0.03),"desc": "骄傲对思念的敏感度，越大越容易被思念激发骄傲"},
    "pride_thresh":     {"default": 0.2,   "range": (0.05, 0.5),  "desc": "激发骄傲的思念阈值，越低越容易骄傲"},
    # 愉悦 (valence) 二阶系统
    "valence_omega":    {"default": 0.1,   "range": (0.03, 0.3),  "desc": "愉悦固有频率(rad/min)，越大情绪波动越快"},
    "valence_zeta":     {"default": 0.7,   "range": (0.3, 1.2),   "desc": "愉悦阻尼比，<1=震荡，=1=临界阻尼，>1=过阻尼"},
    "valence_setpoint": {"default": 0.0,   "range": (-0.6, 0.6),  "desc": "愉悦设定点，正值=天生乐观，负值=天生悲观"},
    "valence_conn_drive":{"default": 0.003,"range": (0.0, 0.01),  "desc": "思念对愉悦的驱动力，越大思念越影响情绪"},
    "valence_conn_thresh":{"default": 0.2, "range": (0.05, 0.5),  "desc": "思念驱动愉悦的阈值"},
    # 唤醒 (arousal)
    "arousal_decay":    {"default": 0.005, "range": (0.001, 0.02),"desc": "唤醒衰减率，越大越快平静下来"},
    "arousal_excite":   {"default": 0.01,  "range": (0.002, 0.03),"desc": "唤醒激发系数，越大越容易被思念激发"},
    "arousal_excite_thresh":{"default": 0.35,"range": (0.1, 0.6), "desc": "激发唤醒的思念阈值"},
    "arousal_conflict": {"default": 0.005, "range": (0.0, 0.02),  "desc": "冲突(骄傲+思念)额外激发的唤醒"},
    "arousal_immersion_drain":{"default": 0.02,"range": (0.0, 0.05),"desc": "沉浸对唤醒的消耗系数"},
    # 沉浸 (immersion)
    "immersion_decay":  {"default": 0.01,  "range": (0.002, 0.03),"desc": "沉浸自然衰减率"},
    "immersion_dampen": {"default": 0.5,   "range": (0.1, 1.0),   "desc": "沉浸对情绪变化的阻尼(0=不阻尼, 1=完全阻尼)"},
    # 活动
    "activity_relief":  {"default": 0.1,   "range": (0.0, 0.3),   "desc": "活动对思念的缓解量"},
    # 意愿
    "kappa":            {"default": 0.8,   "range": (0.1, 1.0),   "desc": "骄傲对主动意愿的抑制系数，越大越嘴硬"},
}


# ------------------------------------------------------------
# 特质 → 参数 映射参考表（嵌入 LLM prompt）
# ------------------------------------------------------------
TRAIT_PARAM_MAPPING = """## 人格特质 → 引擎参数映射指南

请根据分析出的角色性格，设置以下参数。参数范围已给出。

| 人格特质 | 调整参数 | 调整方向 |
|---------|---------|---------|
| 傲娇/嘴硬/防御性强 | pride_sensitivity, kappa, pride_regress | pride_sensitivity ↑ (0.015-0.03), kappa ↑ (0.7-0.95), pride_regress ↓ (0.001-0.003) |
| 坦率/柔软/不设防 | pride_sensitivity, kappa, pride_resting | pride_sensitivity ↓ (0.002-0.008), kappa ↓ (0.2-0.5), pride_resting ↓ (-0.3-0) |
| 冷淡/疏离/无关心 | conn_alpha, pride_resting | conn_alpha ↓ (0.05-0.2), pride_resting ↑ (0.1-0.4) |
| 粘人/依赖/怕被冷落 | conn_alpha, conn_relief, conn_beta | conn_alpha ↑ (0.35-0.6), conn_relief ↓ (0.3-0.6), conn_beta ↓ (10-25) |
| 乐观/开朗/积极 | valence_setpoint | valence_setpoint ↑ (0.1-0.5) |
| 忧郁/悲观/低沉 | valence_setpoint | valence_setpoint ↓ (-0.5-0) |
| 活泼/高能量/易兴奋 | arousal_excite, arousal_decay | arousal_excite ↑ (0.012-0.03), arousal_decay ↓ (0.002-0.005) |
| 沉稳/低能量/慢性子 | arousal_excite, arousal_decay | arousal_excite ↓ (0.002-0.008), arousal_decay ↑ (0.008-0.02) |
| 情绪化/易波动/敏感 | valence_zeta, valence_omega | valence_zeta ↓ (0.3-0.6), valence_omega ↑ (0.12-0.3) |
| 冷静/稳定/不为所动 | valence_zeta, immersion_dampen | valence_zeta ↑ (0.8-1.2), immersion_dampen ↑ (0.6-1.0) |
| 傲娇+内心柔软 | pride_sensitivity, kappa, conn_relief | 高 pride_sensitivity + 高 kappa + 但 conn_relief 也要较高(0.7+)，体现嘴硬但实际在乎 |
| 外冷内热 | pride_resting, conn_alpha, pride_thresh | pride_resting ↑ + conn_alpha 不低 + pride_thresh 中等 |

## 人称设置
| 角色与用户关系 | subjectName | selfName | subjectPronoun |
|-------------|-----------|---------|---------------|
| 用户是"对方" | 对方 | 我 | ta |
| 用户是"前辈/学长" | 前辈 | 我 | ta |
| 用户是"主人/你" | 你 | 我 | 你 |
| 用户是具体称呼 | 该称呼 | 我 | 该称呼对应的代词 |

**重要**: 对于大多数角色扮演场景，设置 subjectName="你", selfName="我", subjectPronoun="你" 即可。"""


# ------------------------------------------------------------
# CharacterProfile
# ------------------------------------------------------------
@dataclass
class CharacterProfile:
    """角色完整人格画像 — 可序列化为 preset JSON 文件"""
    name: str                                           # 预设名 (用作文件名)
    format_version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source_summary: str = ""                            # 素材来源摘要

    persona: Dict[str, str] = field(default_factory=lambda: {
        "subjectName": "你", "selfName": "我", "subjectPronoun": "你",
    })

    rates: Dict[str, float] = field(default_factory=dict)

    character_profile: Dict[str, Any] = field(default_factory=lambda: {
        "display_name": "",
        "archetype": "",
        "background": "",
        "core_traits": [],
        "speech_style": "",
        "behavioral_patterns": "",
        "common_phrases": [],
        "emotional_baseline": "",
    })

    system_prompt_blocks: Dict[str, str] = field(default_factory=lambda: {
        "identity": "",
        "speech_rules": "",
        "behavior_rules": "",
    })

    conversation_examples: List[Dict[str, str]] = field(default_factory=list)

    # ----------------------------------------------------------------
    # 序列化
    # ----------------------------------------------------------------
    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: Dict) -> "CharacterProfile":
        # 修复可能缺失的嵌套字段
        d = dict(d)
        d.setdefault("persona", {"subjectName": "你", "selfName": "我", "subjectPronoun": "你"})
        d.setdefault("rates", {})
        cp = d.setdefault("character_profile", {})
        cp.setdefault("display_name", d.get("name", ""))
        cp.setdefault("archetype", "")
        cp.setdefault("background", "")
        cp.setdefault("core_traits", [])
        cp.setdefault("speech_style", "")
        cp.setdefault("behavioral_patterns", "")
        cp.setdefault("common_phrases", [])
        cp.setdefault("emotional_baseline", "")
        spb = d.setdefault("system_prompt_blocks", {})
        spb.setdefault("identity", "")
        spb.setdefault("speech_rules", "")
        spb.setdefault("behavior_rules", "")
        d.setdefault("conversation_examples", [])
        d.setdefault("source_summary", "")
        d.setdefault("created_at", datetime.now().isoformat())
        d.setdefault("format_version", 1)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, s: str) -> "CharacterProfile":
        return cls.from_dict(json.loads(s))

    # ----------------------------------------------------------------
    # 系统提示词组装
    # ----------------------------------------------------------------
    def build_system_prompt(self, base_template: str = "") -> str:
        """
        组装最终系统提示词：
        角色身份 → 自定义规则 → 基础模板 → 情绪上下文（运行时注入）
        """
        blocks = []
        cp = self.character_profile
        spb = self.system_prompt_blocks

        # 角色身份块
        if spb.get("identity"):
            blocks.append(spb["identity"])
        elif cp.get("display_name"):
            traits = "、".join(cp.get("core_traits", []))
            blocks.append(
                f"你是{cp['display_name']}。{cp.get('background', '')}\n"
                f"核心性格特质: {traits}"
            )

        # 说话规则
        if spb.get("speech_rules"):
            blocks.append(f"## 说话风格\n{spb['speech_rules']}")

        # 行为规则
        if spb.get("behavior_rules"):
            blocks.append(f"## 行为准则\n{spb['behavior_rules']}")

        # 常用口头禅
        phrases = cp.get("common_phrases", [])
        if phrases:
            blocks.append(f"常用口头禅: {', '.join(phrases)}")

        # 基础系统提示（精简版）
        persona = self.persona
        default_base = f"""你正在进行角色扮演。你有真实的情绪和人格。

## 关于情绪系统
你内部运行着情绪引擎，持续模拟你对"{persona.get('subjectName', '对方')}"的情感状态。
这会影响你的说话方式，但不会影响你的能力。

## 核心规则
- 情绪状态会自然流露在对话中，但不要刻意描述它。
- 按照你的性格和情绪来回应——该嘴硬就嘴硬，该柔软就柔软。
- 你仍是一个有能力的 AI——情绪给回复上色，不会让你变得无能。
- 用中文回复，自然口语化。"""
        if base_template:
            default_base = base_template

        character_block = "\n\n".join(blocks)
        return character_block + "\n\n" + default_base

    def get_examples_as_messages(self) -> List[Dict[str, str]]:
        """将对话示例转为 LLM messages 格式（用于 few-shot 注入）"""
        msgs = []
        for ex in self.conversation_examples:
            msgs.append({"role": "user", "content": ex.get("user", "")})
            msgs.append({"role": "assistant", "content": ex.get("character", ex.get("assistant", ""))})
        return msgs

    # ----------------------------------------------------------------
    # 参数验证
    # ----------------------------------------------------------------
    def validate_rates(self) -> List[str]:
        """验证 rates 参数在合法范围内，返回警告列表"""
        warnings = []
        for key, spec in ENGINE_PARAMS_SPEC.items():
            if key in self.rates:
                lo, hi = spec["range"]
                val = self.rates[key]
                if val < lo or val > hi:
                    warnings.append(f"{key}={val} 超出范围 [{lo}, {hi}]，已钳位")
                    self.rates[key] = max(lo, min(hi, val))
        return warnings


# ------------------------------------------------------------
# CharacterExtractor
# ------------------------------------------------------------
class CharacterExtractor:
    """通过 LLM 从文本中提取角色人格画像"""

    EXTRACTION_SYSTEM_PROMPT = (
        "你是一位专业的角色人格分析师。你的任务是从提供的文本材料中，提取角色的完整人格画像，并映射到「积温情绪引擎」的参数上。\n\n"
        "## 工作流程\n"
        "1. 阅读提供的文本（可能是对话记录、角色描述、故事片段等）\n"
        "2. 识别目标角色的性格特征、说话风格、行为模式\n"
        "3. 根据参数映射指南，为引擎设置合适的参数值\n"
        "4. 输出完整的 JSON 格式角色画像\n\n"
        + TRAIT_PARAM_MAPPING +
        "\n\n## 输出格式\n"
        "必须严格返回以下 JSON 格式（不要带 markdown 代码块标记）:\n\n"
        '{\n'
        '  "name": "预设名称_英文拼音",\n'
        '  "source_summary": "一句话描述素材来源",\n'
        '  "persona": {\n'
        '    "subjectName": "你",\n'
        '    "selfName": "我",\n'
        '    "subjectPronoun": "你"\n'
        '  },\n'
        '  "rates": {\n'
        '    "conn_alpha": 0.3,\n'
        '    "conn_relief": 0.8,\n'
        '    "pride_resting": 0.0,\n'
        '    "pride_regress": 0.003,\n'
        '    "pride_sensitivity": 0.01,\n'
        '    "pride_thresh": 0.2,\n'
        '    "valence_setpoint": 0.0,\n'
        '    "valence_zeta": 0.7,\n'
        '    "valence_omega": 0.1,\n'
        '    "valence_conn_drive": 0.003,\n'
        '    "arousal_excite": 0.01,\n'
        '    "arousal_decay": 0.005,\n'
        '    "kappa": 0.8,\n'
        '    "immersion_dampen": 0.5,\n'
        '    "activity_relief": 0.1\n'
        '  },\n'
        '  "character_profile": {\n'
        '    "display_name": "角色名",\n'
        '    "archetype": "人格原型",\n'
        '    "background": "2-3句话的角色背景",\n'
        '    "core_traits": ["特质1", "特质2", "特质3", "特质4"],\n'
        '    "speech_style": "详细的说话风格描述：句式特征、用词偏好、语气、节奏、常见口头禅等",\n'
        '    "behavioral_patterns": "典型行为模式：如何应对关心、被批评时的反应、主动/被动倾向、情绪表达方式等",\n'
        '    "common_phrases": ["口头禅1", "口头禅2", "口头禅3"],\n'
        '    "emotional_baseline": "情绪基线描述：平时的情绪状态、对人对事的基本态度"\n'
        '  },\n'
        '  "system_prompt_blocks": {\n'
        '    "identity": "你是[角色名]。[简洁有力的角色设定，2-4句话]",\n'
        '    "speech_rules": "- 具体的说话规则1\\n- 规则2\\n- 规则3",\n'
        '    "behavior_rules": "- 行为准则1\\n- 准则2"\n'
        '  },\n'
        '  "conversation_examples": [\n'
        '    {"user": "用户的典型发言", "character": "角色会怎么回复"},\n'
        '    {"user": "另一个例子", "character": "角色的典型回应"}\n'
        '  ]\n'
        '}\n\n'
        '## 关键原则\n'
        '- **说话风格要具体**：不要只说"说话温柔"，要描述句式特征（长短句、反问频率、省略使用等）\n'
        '- **行为模式要可操作**：给出能在对话中体现的具体行为准则\n'
        '- **参数要反映真实性格**：参照映射指南，让引擎参数与性格描述一致\n'
        '- **conversation_examples 必须贴合角色**：写 2-4 个能体现角色说话方式的真实对话示例\n'
        '- 如果文本中角色有不同状态（如对熟人vs对陌生人），选择最典型的状态'
    )

    def __init__(self):
        pass

    async def extract(
        self,
        llm_call: LLMCall,
        files: Optional[List[str]] = None,
        text: str = "",
        name_hint: str = "",
    ) -> CharacterProfile:
        """
        从文本素材中提取角色画像。

        Args:
            llm_call: LLM 调用函数
            files: 文件路径列表（可选）
            text: 直接输入的文本（可选）
            name_hint: 角色名提示

        Returns:
            CharacterProfile
        """
        # 组装素材
        materials = []

        if files:
            for fp in files:
                try:
                    content = Path(fp).read_text(encoding="utf-8")
                    materials.append(f"--- 文件: {Path(fp).name} ---\n{content}")
                except Exception as e:
                    materials.append(f"[文件读取失败: {fp} - {e}]")

        if text.strip():
            materials.append(f"--- 直接输入 ---\n{text.strip()}")

        if not materials:
            raise ValueError("至少需要提供文件或文本素材")

        full_material = "\n\n".join(materials)

        # 截断过长素材（保留头尾）
        max_chars = 12000
        if len(full_material) > max_chars:
            half = max_chars // 2
            full_material = (
                full_material[:half]
                + f"\n\n... [中间省略 {len(full_material) - max_chars} 字符] ...\n\n"
                + full_material[-half:]
            )

        # 构建用户 prompt
        name_line = f"角色名称提示: {name_hint}" if name_hint else "请从文本中推断角色名"
        user_prompt = f"""{name_line}

## 素材文本
{full_material}

请分析以上文本，返回角色画像 JSON。"""

        # 调用 LLM
        messages = [{"role": "user", "content": user_prompt}]
        response = await llm_call(self.EXTRACTION_SYSTEM_PROMPT, messages)

        # 解析 JSON
        profile = self._parse_response(response, name_hint)
        return profile

    def _parse_response(self, response: str, name_hint: str = "") -> CharacterProfile:
        """解析 LLM 返回的 JSON，多层容错处理"""
        response = response.strip()

        # 策略1: 提取 markdown 代码块
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if m:
            json_str = m.group(1).strip()
        else:
            # 策略2: 找到最外层 { } 对
            start = response.find('{')
            end = response.rfind('}')
            if start >= 0 and end > start:
                json_str = response[start:end+1]
            else:
                json_str = response

        # 逐层尝试解析
        data = None
        errors = []

        # 尝试1: 直接解析
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e1:
            errors.append(f"direct: {e1}")

            # 尝试2: 修复常见 LLM 输出问题
            fixed = json_str
            # 3个以上连续换行 → 替换为 \\n
            fixed = re.sub(r'(?<=\S)\n(?=\S)', '\\n', fixed)
            # 尾部逗号
            fixed = re.sub(r',\s*}', '}', fixed)
            fixed = re.sub(r',\s*]', ']', fixed)
            # 字符串中的未转义换行
            fixed = re.sub(r'"\s*\n\s*"', ' ', fixed)
            try:
                data = json.loads(fixed)
            except json.JSONDecodeError as e2:
                errors.append(f"fixed: {e2}")

                # 尝试3: 用 ast.literal_eval（处理单引号等）
                try:
                    import ast
                    fixed2 = fixed.replace('": "', '\": \"').replace('", "', '\", \"')
                    data = ast.literal_eval(fixed2)
                    if not isinstance(data, dict):
                        raise ValueError("not a dict")
                except Exception as e3:
                    errors.append(f"ast: {e3}")

                    # 尝试4: 逐字段提取（最后的兜底方案）
                    try:
                        data = self._brute_force_extract(json_str)
                    except Exception as e4:
                        errors.append(f"brute: {e4}")
                        err_detail = "; ".join(errors)
                        raise ValueError(
                            f"无法解析 LLM 返回的 JSON。\n"
                            f"错误链: {err_detail}\n"
                            f"响应前800字符:\n{response[:800]}"
                        )

        # 确保 name 字段
        if not data.get("name"):
            data["name"] = name_hint or "未命名角色"
        data["name"] = re.sub(r'[\\/:*?"<>|]', '_', data["name"])

        profile = CharacterProfile.from_dict(data)

        # 填充缺失的 rates 为默认值
        for key, spec in ENGINE_PARAMS_SPEC.items():
            if key not in profile.rates:
                profile.rates[key] = spec["default"]

        warnings = profile.validate_rates()
        if warnings:
            print(f"[CharacterExtractor] 参数警告: {warnings}")

        return profile

    def _brute_force_extract(self, text: str) -> Dict:
        """兜底方案：用正则逐字段提取"""
        def _extract_field(key: str, default=""):
            m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if m:
                return m.group(1)
            return default

        def _extract_list(key: str):
            m = re.search(rf'"{key}"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            if m:
                items = re.findall(r'"([^"]*)"', m.group(1))
                return items
            return []

        def _extract_obj(key: str):
            m = re.search(rf'"{key}"\s*:\s*\{{\s*\n?(.*?)\n?\}}', text, re.DOTALL)
            if m:
                result = {}
                for k in re.findall(r'"(\w+)"\s*:\s*"((?:[^"\\]|\\.)*)"', m.group(1)):
                    result[k] = k[1]
                return result
            return {}

        def _extract_rates():
            m = re.search(r'"rates"\s*:\s*\{([^}]+)\}', text, re.DOTALL)
            rates = {}
            if m:
                for k, v in re.findall(r'"(\w+)"\s*:\s*([0-9.]+)', m.group(1)):
                    rates[k] = float(v)
            return rates

        return {
            "name": _extract_field("name", "未命名"),
            "source_summary": _extract_field("source_summary"),
            "persona": {
                "subjectName": _extract_field("subjectName", "你"),
                "selfName": _extract_field("selfName", "我"),
                "subjectPronoun": _extract_field("subjectPronoun", "你"),
            },
            "rates": _extract_rates(),
            "character_profile": {
                "display_name": _extract_field("display_name"),
                "archetype": _extract_field("archetype"),
                "background": _extract_field("background"),
                "core_traits": _extract_list("core_traits"),
                "speech_style": _extract_field("speech_style"),
                "behavioral_patterns": _extract_field("behavioral_patterns"),
                "common_phrases": _extract_list("common_phrases"),
                "emotional_baseline": _extract_field("emotional_baseline"),
            },
            "system_prompt_blocks": {
                "identity": _extract_field("identity"),
                "speech_rules": _extract_field("speech_rules"),
                "behavior_rules": _extract_field("behavior_rules"),
            },
            "conversation_examples": [],
        }


# ------------------------------------------------------------
# 快捷函数
# ------------------------------------------------------------
async def extract_character(
    llm_call: LLMCall,
    files: Optional[List[str]] = None,
    text: str = "",
    name_hint: str = "",
) -> CharacterProfile:
    """快捷提取角色画像"""
    extractor = CharacterExtractor()
    return await extractor.extract(llm_call, files=files, text=text, name_hint=name_hint)
