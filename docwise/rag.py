"""检索增强问答：文本切分、BM25 检索、历史问题改写与流式回答。"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import jieba
from langchain.chains import create_history_aware_retriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Plus

from docwise.loaders import source_label

jieba.setLogLevel(logging.WARNING)

# 中文文档常见的分段边界，优先在段落和句号处切分。
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

# 高频虚词不参与检索，避免小语料下公共词干扰排序。
STOPWORDS = frozenset(
    """
    的 了 和 与 及 或 是 在 有 为 对 从 到 把 被 让 给 向 于 而 就 都 也
    还 很 更 最 又 再 才 只 不 没 无 我 你 他 她 它 我们 你们 他们 这 那
    这个 那个 这些 那些 什么 怎么 怎样 如何 哪些 哪个 为什么 请问
    可以 可能 能够 需要 应该 一个 一些 一种 进行 通过 以及 但是 因为
    所以 如果 虽然 然后 已经 还是 就是 还有 并且 或者 等等 之 其 以 由
    则 并 且 等 中 上 下 里 内 外 前 后 时 地 得 着 过 吗 呢 吧 啊 呀
    哦
    """.split()
)

QA_SYSTEM_PROMPT = """你是智阅 DocWise，一位严谨、友好的文档阅读助手。请严格依据下面提供的资料回答用户问题。

回答要求：
1. 只使用资料中的信息，不要编造资料中没有的内容。
2. 每个结论后用 [片段 1] 这样的编号标明依据，可以同时引用多个片段。
3. 如果资料不足以回答，请直接说明缺少什么信息，并给出资料中能够确认的部分。
4. 使用简体中文，条理清晰，可以用 Markdown 列表或小标题组织答案。
5. 回答保持简洁，除非用户要求详细展开。

资料如下：
{context}"""

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "给定下面的对话历史和用户的新问题，请把新问题改写成一个不依赖上下文、"
            "语义完整的独立问题。只输出改写后的问题，不要回答问题，也不要添加解释。",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是智阅 DocWise，一个友善、务实的 AI 助手。"
            "请用简体中文回答，表达清晰，必要时使用 Markdown 排版。",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def split_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[Document]:
    """按中文标点递归切分长文档。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk"] = index
    return [chunk for chunk in chunks if chunk.page_content.strip()]


def _tokenize(text: str) -> list[str]:
    """BM25 使用 jieba 搜索模式分词，兼顾中文词语与英文关键词。"""
    tokens: list[str] = []
    for raw in jieba.lcut_for_search(text):
        token = raw.strip().lower()
        if not token or token in STOPWORDS:
            continue
        if not any(char.isalnum() for char in token):
            continue
        tokens.append(token)
    return tokens


def build_retriever(chunks: list[Document], k: int = 4) -> BM25Retriever:
    """构建 BM25 关键词检索器，无需额外下载向量模型，可完全离线运行。

    这里直接注入 BM25Plus 评分器。相比 LangChain 默认的 BM25Okapi，
    BM25Plus 的逆文档频率恒为正数，在小语料下排序更稳定。
    """
    if not chunks:
        raise ValueError("没有可检索的文档片段。")

    vectorizer = BM25Plus([_tokenize(chunk.page_content) for chunk in chunks])
    return BM25Retriever(
        vectorizer=vectorizer,
        docs=list(chunks),
        k=max(1, k),
        preprocess_func=_tokenize,
    )


def format_docs(documents: list[Document]) -> str:
    """把检索结果拼成带编号和来源的上下文。"""
    blocks = []
    for index, document in enumerate(documents, start=1):
        blocks.append(
            f"【片段 {index}｜{source_label(document)}】\n{document.page_content}"
        )
    return "\n\n".join(blocks)


def retrieve_context(
    question: str,
    llm,
    retriever: BM25Retriever,
    history: list[BaseMessage],
) -> list[Document]:
    """检索相关片段。存在多轮对话时，先用大模型把追问改写成独立问题。"""
    if not history:
        return retriever.invoke(question)

    chain = create_history_aware_retriever(llm, retriever, CONDENSE_PROMPT)
    return chain.invoke({"input": question, "chat_history": history})


def stream_qa_answer(
    question: str,
    llm,
    documents: list[Document],
    history: list[BaseMessage],
) -> Iterator[str]:
    """基于检索到的片段流式生成回答。"""
    chain = QA_PROMPT | llm | StrOutputParser()
    yield from chain.stream(
        {
            "context": format_docs(documents),
            "chat_history": history,
            "input": question,
        }
    )


def stream_chat(
    question: str,
    llm,
    history: list[BaseMessage],
) -> Iterator[str]:
    """不依赖文档的自由对话。"""
    chain = CHAT_PROMPT | llm | StrOutputParser()
    yield from chain.stream({"chat_history": history, "input": question})
