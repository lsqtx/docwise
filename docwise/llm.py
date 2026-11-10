"""大模型工厂：统一创建在线 OpenAI 兼容模型与本地 Ollama 模型。"""

from __future__ import annotations

import os

from docwise.config import PROVIDERS


class MissingAPIKeyError(ValueError):
    """没有提供可用的 API Key。"""


def create_llm(
    provider: str,
    model: str = "",
    api_key: str = "",
    base_url: str = "",
    temperature: float = 0.2,
):
    """根据服务商创建 LangChain ChatModel。

    在线服务统一走 OpenAI 兼容协议，本地服务走 Ollama。
    参数中的空字符串会回退到预设值或环境变量。
    """
    if provider not in PROVIDERS:
        raise ValueError(f"未知的模型服务：{provider}")

    preset = PROVIDERS[provider]
    resolved_base = (base_url or preset["base_url"]).strip()
    resolved_model = (model or preset["model"]).strip()

    if preset["kind"] == "offline":
        if not resolved_base:
            raise ValueError("请填写 Ollama 服务地址，例如 http://localhost:11434")
        if not resolved_model:
            raise ValueError("请填写 Ollama 模型名称，例如 qwen2.5:7b")
        # 延迟导入，未安装 Ollama 依赖时不影响在线模型的启动。
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=resolved_model,
            base_url=resolved_base,
            temperature=temperature,
        )

    env_name = preset.get("key_env", "")
    resolved_key = (api_key or (os.getenv(env_name, "") if env_name else "")).strip()
    if not resolved_key:
        hint = f"，可在 {preset['signup']} 申请" if preset.get("signup") else ""
        raise MissingAPIKeyError(f"缺少 {provider} 的 API Key{hint}。")
    if not resolved_base:
        raise ValueError("请填写接口地址 Base URL。")
    if not resolved_model:
        raise ValueError("请填写模型名称。")

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=resolved_model,
        api_key=resolved_key,
        base_url=resolved_base,
        temperature=temperature,
        timeout=60,
        max_retries=2,
    )
