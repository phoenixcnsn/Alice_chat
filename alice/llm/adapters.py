"""
LLM 适配器工厂 — Anthropic / OpenAI / DeepSeek

纯文本接口: LLMCall = (system_prompt, messages) -> str
工具调用由 agent 层通过文本解析实现，不依赖 API function calling。
"""

from typing import Any, Awaitable, Callable, Dict, List, Optional

LLMCall = Callable[[str, List[Dict[str, str]]], Awaitable[str]]


# ------------------------------------------------------------
# OpenAI / DeepSeek
# ------------------------------------------------------------

async def create_openai_compatible_adapter(
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    max_tokens: int = 1024,
    temperature: float = 0.9,
    base_url: str = "",
) -> LLMCall:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError("需要安装 openai SDK: pip install openai")

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = AsyncOpenAI(**kwargs)

    async def call(system_prompt: str, messages: List[Dict[str, str]]) -> str:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response = await client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=full_messages,
        )
        return response.choices[0].message.content or ""

    return call


# ------------------------------------------------------------
# Anthropic
# ------------------------------------------------------------

async def create_anthropic_adapter(
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 1024,
    temperature: float = 0.9,
    base_url: str = "",
) -> LLMCall:
    try:
        import anthropic
    except ImportError:
        raise ImportError("需要安装 anthropic SDK: pip install anthropic")

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = anthropic.AsyncAnthropic(**kwargs)

    async def call(system_prompt: str, messages: List[Dict[str, str]]) -> str:
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text

    return call


# ------------------------------------------------------------
# 便捷工厂
# ------------------------------------------------------------

async def create_openai_adapter(**kwargs) -> LLMCall:
    return await create_openai_compatible_adapter(**kwargs)


async def create_deepseek_adapter(**kwargs) -> LLMCall:
    if not kwargs.get("base_url"):
        kwargs["base_url"] = "https://api.deepseek.com/v1"
    if not kwargs.get("model"):
        kwargs["model"] = "deepseek-chat"
    return await create_openai_compatible_adapter(**kwargs)
