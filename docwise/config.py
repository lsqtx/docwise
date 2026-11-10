"""智阅 DocWise 的配置与模型服务预设。"""

from __future__ import annotations

import os

APP_NAME = "智阅 DocWise"
APP_TAGLINE = "基于 LangChain 的智能文档问答助手"
VERSION = "1.0.0"

# 每个预设描述一个 OpenAI 兼容的在线服务，或本地 Ollama 服务。
# kind: online 表示需要 API Key，offline 表示本地推理。
PROVIDERS: dict[str, dict[str, str]] = {
    "DeepSeek 深度求索": {
        "kind": "online",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "key_env": "DEEPSEEK_API_KEY",
        "signup": "https://platform.deepseek.com",
        "note": "国内直连，性价比高，推荐使用 deepseek-chat。",
    },
    "阿里云百炼（通义千问）": {
        "kind": "online",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "key_env": "DASHSCOPE_API_KEY",
        "signup": "https://bailian.console.aliyun.com",
        "note": "阿里云百炼控制台创建 API Key，兼容 OpenAI 协议。",
    },
    "智谱 GLM": {
        "kind": "online",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
        "key_env": "ZHIPUAI_API_KEY",
        "signup": "https://open.bigmodel.cn",
        "note": "glm-4-flash 提供免费额度，适合课程演示。",
    },
    "月之暗面 Kimi": {
        "kind": "online",
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
        "key_env": "MOONSHOT_API_KEY",
        "signup": "https://platform.moonshot.cn",
        "note": "长上下文能力较强，适合长文档阅读。",
    },
    "OpenAI 兼容自定义": {
        "kind": "online",
        "base_url": "",
        "model": "",
        "key_env": "OPENAI_API_KEY",
        "signup": "",
        "note": "填写任意兼容 OpenAI 协议的接口地址与模型名称。",
    },
    "Ollama 本地模型": {
        "kind": "offline",
        "base_url": "http://localhost:11434",
        "model": "qwen2.5:7b",
        "key_env": "",
        "signup": "https://ollama.com",
        "note": "完全离线运行，先执行 ollama pull qwen2.5:7b 拉取模型。",
    },
}


def default_provider() -> str:
    """优先选择已经配置了环境变量的在线服务，否则回退到 DeepSeek。"""
    for name, preset in PROVIDERS.items():
        env_name = preset.get("key_env", "")
        if preset["kind"] == "online" and env_name and os.getenv(env_name):
            return name
    return "DeepSeek 深度求索"
