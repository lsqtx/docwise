"""智阅 DocWise 的 Streamlit 界面入口。"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from docwise.config import APP_NAME, APP_TAGLINE, PROVIDERS, VERSION, default_provider
from docwise.llm import create_llm
from docwise.loaders import SUPPORTED_SUFFIXES, load_document, source_label
from docwise.rag import (
    build_retriever,
    retrieve_context,
    split_documents,
    stream_chat,
    stream_qa_answer,
)

load_dotenv(Path(__file__).with_name(".env"))

st.set_page_config(
    page_title=f"{APP_NAME} · 智能文档问答",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1080px; }
    .docwise-hero {
        padding: 1.4rem 1.7rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #0f766e 0%, #0e7490 55%, #1d4ed8 100%);
        color: #ffffff;
        margin-bottom: 1rem;
        box-shadow: 0 12px 30px rgba(15, 118, 110, 0.18);
    }
    .docwise-title { font-size: 2rem; font-weight: 800; letter-spacing: 0.04em; }
    .docwise-sub { font-size: 0.95rem; opacity: 0.93; margin-top: 0.35rem; }
    .docwise-meta { font-size: 0.76rem; opacity: 0.75; margin-top: 0.55rem; }
    .docwise-status {
        display: inline-block;
        padding: 0.28rem 0.7rem;
        margin: 0 0.4rem 0.6rem 0;
        border-radius: 999px;
        background: rgba(15, 118, 110, 0.10);
        color: #0f766e;
        font-size: 0.78rem;
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1rem; }
    div[data-testid="stChatMessage"] { border-radius: 14px; }
    [data-testid="stToolbar"], [data-testid="stAppDeployButton"] { display: none; }
    #MainMenu, footer { visibility: hidden; }
</style>
"""

UPLOAD_TYPES = sorted(suffix.lstrip(".") for suffix in SUPPORTED_SUFFIXES)


def init_state() -> None:
    if "provider" in st.session_state:
        return

    provider = default_provider()
    preset = PROVIDERS[provider]
    st.session_state.provider = provider
    st.session_state.base_url_input = preset["base_url"]
    st.session_state.model_input = preset["model"]
    st.session_state.api_key_input = (
        os.getenv(preset["key_env"], "") if preset.get("key_env") else ""
    )
    st.session_state.temperature = 0.2
    st.session_state.retriever_k = 4
    st.session_state.qa_messages = []
    st.session_state.chat_messages = []
    st.session_state.doc_chunks = []
    st.session_state.doc_sig = ()
    st.session_state.retriever = None
    st.session_state.retriever_key = None


def on_provider_change() -> None:
    preset = PROVIDERS[st.session_state.provider]
    st.session_state.base_url_input = preset["base_url"]
    st.session_state.model_input = preset["model"]
    st.session_state.api_key_input = (
        os.getenv(preset["key_env"], "") if preset.get("key_env") else ""
    )
    st.session_state.retriever = None
    st.session_state.retriever_key = None


def on_retriever_k_change() -> None:
    st.session_state.retriever = None
    st.session_state.retriever_key = None


def build_llm():
    return create_llm(
        provider=st.session_state.provider,
        model=st.session_state.get("model_input", ""),
        api_key=st.session_state.get("api_key_input", ""),
        base_url=st.session_state.get("base_url_input", ""),
        temperature=st.session_state.get("temperature", 0.2),
    )


def get_retriever():
    chunks = st.session_state.doc_chunks
    k = st.session_state.retriever_k
    cache_key = (st.session_state.doc_sig, len(chunks), k)
    if st.session_state.retriever is None or st.session_state.retriever_key != cache_key:
        st.session_state.retriever = build_retriever(chunks, k=k)
        st.session_state.retriever_key = cache_key
    return st.session_state.retriever


def to_messages(messages: list[dict], limit: int = 8) -> list[BaseMessage]:
    result: list[BaseMessage] = []
    for message in messages[-limit:]:
        cls = HumanMessage if message["role"] == "user" else AIMessage
        result.append(cls(content=message["content"]))
    return result


def render_sources(documents: list) -> None:
    with st.expander(f"查看引用来源（{len(documents)} 个片段）", expanded=False):
        for index, document in enumerate(documents, start=1):
            st.markdown(f"**片段 {index}｜{source_label(document)}**")
            preview = document.page_content.strip().replace("\n", " ")
            if len(preview) > 450:
                preview = preview[:450] + "……"
            st.markdown(f"> {preview}")


def stream_to_ui(generator) -> str:
    placeholder = st.empty()
    parts: list[str] = []
    for token in generator:
        parts.append(token)
        placeholder.markdown("".join(parts) + " ▌")
    text = "".join(parts)
    placeholder.markdown(text)
    return text


def index_uploads(uploads, signature: tuple) -> None:
    documents = []
    failures: list[str] = []
    with st.spinner("正在解析并切分文档……"):
        for item in uploads:
            try:
                documents.extend(load_document(item.name, item.getvalue()))
            except Exception as exc:  # 单个文件失败不影响其他文件
                failures.append(f"{item.name}：{exc}")
        chunks = split_documents(documents) if documents else []

    st.session_state.doc_sig = signature
    st.session_state.doc_chunks = chunks
    st.session_state.retriever = None
    st.session_state.retriever_key = None
    st.session_state.qa_messages = []

    if chunks:
        st.success(f"已索引 {len(uploads) - len(failures)} 个文件，共 {len(chunks)} 个片段。")
    else:
        st.warning("没有从文件中解析出可用文本，请检查文件内容或换用其他文件。")
    for failure in failures:
        st.warning(f"解析失败：{failure}")


def conversation_markdown() -> str:
    lines = [
        f"# {APP_NAME} 对话记录",
        "",
        f"导出时间：{datetime.now():%Y-%m-%d %H:%M}",
        "",
    ]
    for title, messages in (
        ("文档问答", st.session_state.qa_messages),
        ("自由对话", st.session_state.chat_messages),
    ):
        if not messages:
            continue
        lines.extend([f"## {title}", ""])
        for message in messages:
            role = "我" if message["role"] == "user" else "DocWise"
            lines.append(f"**{role}**：{message['content']}")
            lines.append("")
    return "\n".join(lines)


def render_sidebar() -> tuple:
    with st.sidebar:
        st.markdown(f"### 📚 {APP_NAME}")
        st.caption(APP_TAGLINE)
        st.divider()

        st.markdown("#### 模型设置")
        provider = st.selectbox(
            "模型服务",
            list(PROVIDERS),
            key="provider",
            on_change=on_provider_change,
        )
        preset = PROVIDERS[provider]

        if preset["kind"] == "online":
            st.text_input(
                "API Key",
                key="api_key_input",
                type="password",
                help="仅保存在当前浏览器会话，也可以写入 .env 文件。",
            )

        st.text_input("接口地址 Base URL", key="base_url_input")
        st.text_input("模型名称", key="model_input")
        st.slider(
            "生成温度",
            0.0,
            1.0,
            key="temperature",
            step=0.05,
            help="数值越低回答越稳定，数值越高表达越灵活。",
        )

        if preset.get("note"):
            st.caption(preset["note"])

        if st.button("测试模型连接", use_container_width=True):
            try:
                with st.spinner("正在连接模型……"):
                    reply = build_llm().invoke("请只回复四个字：连接正常")
                st.success(f"连接成功：{reply.content}")
            except Exception as exc:
                st.error(f"连接失败：{exc}")

        st.divider()

        st.markdown("#### 文档库")
        uploads = st.file_uploader(
            "上传文档",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            key="uploader",
            help="支持 PDF、DOCX、TXT、Markdown，可以一次选择多个文件。",
        )

        if uploads:
            signature = tuple(sorted((item.name, len(item.getvalue())) for item in uploads))
            if signature != st.session_state.doc_sig:
                index_uploads(uploads, signature)
        elif st.session_state.doc_sig:
            st.session_state.doc_sig = ()
            st.session_state.doc_chunks = []
            st.session_state.retriever = None
            st.session_state.retriever_key = None
            st.session_state.qa_messages = []

        st.slider(
            "检索片段数",
            2,
            8,
            key="retriever_k",
            on_change=on_retriever_k_change,
            help="每次问答从文档中检索的相关片段数量。",
        )

        if st.session_state.doc_chunks:
            st.caption(f"当前已索引 {len(st.session_state.doc_chunks)} 个片段。")

        st.divider()

        st.markdown("#### 会话")
        col_export, col_clear = st.columns(2)
        with col_export:
            st.download_button(
                "导出对话",
                data=conversation_markdown(),
                file_name=f"docwise-{datetime.now():%Y%m%d-%H%M}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_clear:
            if st.button("清空对话", use_container_width=True):
                st.session_state.qa_messages = []
                st.session_state.chat_messages = []
                st.rerun()

        st.divider()
        st.caption(f"v{VERSION} · LangChain + Streamlit")
        st.caption("作者：李诗琪 · 软件2405班 · 20245994")

    return provider, preset


def render_header(provider: str, preset: dict) -> None:
    st.markdown(
        f"""
        <div class="docwise-hero">
            <div class="docwise-title">{APP_NAME}</div>
            <div class="docwise-sub">{APP_TAGLINE} · 在线与离线模型自由切换</div>
            <div class="docwise-meta">v{VERSION} · LangChain + Streamlit + BM25</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kind = "离线" if preset["kind"] == "offline" else "在线"
    model = st.session_state.get("model_input") or "未设置"
    chunks = len(st.session_state.doc_chunks)
    st.markdown(
        f'<span class="docwise-status">模型服务：{provider}（{kind}）</span>'
        f'<span class="docwise-status">模型：{model}</span>'
        f'<span class="docwise-status">已索引片段：{chunks}</span>',
        unsafe_allow_html=True,
    )


def render_document_qa() -> None:
    messages = st.session_state.qa_messages
    has_docs = bool(st.session_state.doc_chunks)

    for message in messages:
        avatar = "🧑" if message["role"] == "user" else "📚"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])

    if not messages:
        if has_docs:
            st.info(
                "文档已经就绪，可以开始提问了。例如：这份文档的核心结论是什么？"
                "文中提到了哪些关键数据？"
            )
        else:
            st.info("请先在左侧上传 PDF、DOCX、TXT 或 Markdown 文档，再开始提问。")

    question = st.chat_input(
        "结合文档提问，例如：请总结第三章的主要观点",
        key="qa_input",
        disabled=not has_docs,
    )
    if not question:
        return

    if not st.session_state.doc_chunks:
        st.warning("请先上传文档。")
        return

    history = to_messages(messages)
    st.session_state.qa_messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="📚"):
        try:
            llm = build_llm()
            with st.spinner("正在检索相关片段……"):
                retriever = get_retriever()
                documents = retrieve_context(question, llm, retriever, history)

            if not documents:
                answer = (
                    "没有在已上传的文档中找到相关内容。"
                    "请尝试换一种问法，或者补充相关的文档。"
                )
                st.markdown(answer)
                st.session_state.qa_messages.append(
                    {"role": "assistant", "content": answer, "sources": []}
                )
                return

            answer = stream_to_ui(
                stream_qa_answer(question, llm, documents, history)
            )
            st.session_state.qa_messages.append(
                {"role": "assistant", "content": answer, "sources": documents}
            )
            render_sources(documents)
        except Exception as exc:
            st.error(f"生成失败：{exc}")


def render_free_chat() -> None:
    messages = st.session_state.chat_messages

    for message in messages:
        avatar = "🧑" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if not messages:
        st.info("这里是自由对话模式，不依赖文档，可以直接和模型聊天。")

    question = st.chat_input("输入任意问题，开始对话", key="free_input")
    if not question:
        return

    history = to_messages(messages)
    st.session_state.chat_messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        try:
            llm = build_llm()
            answer = stream_to_ui(stream_chat(question, llm, history))
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": answer}
            )
        except Exception as exc:
            st.error(f"生成失败：{exc}")


def main() -> None:
    init_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    provider, preset = render_sidebar()
    if not st.session_state.get("model_input"):
        st.session_state.model_input = preset["model"]

    render_header(provider, preset)

    mode = st.radio(
        "使用模式",
        ["📄 文档问答", "💬 自由对话"],
        horizontal=True,
        label_visibility="collapsed",
        key="mode",
    )

    if mode.startswith("📄"):
        render_document_qa()
    else:
        render_free_chat()


if __name__ == "__main__":
    main()
