"""不依赖真实大模型的单元测试，覆盖解析、切分、检索与流式链路。"""

from __future__ import annotations

import unittest
from io import BytesIO

from docx import Document as DocxDocument
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage

from docwise.loaders import load_document, source_label
from docwise.rag import (
    _tokenize,
    build_retriever,
    format_docs,
    retrieve_context,
    split_documents,
    stream_qa_answer,
)


def sample_documents() -> list[Document]:
    return [
        Document(
            page_content="智阅 DocWise 是一个基于 LangChain 框架的智能文档问答应用。",
            metadata={"source": "intro.md", "page": 0},
        ),
        Document(
            page_content="系统使用 BM25 与 jieba 分词实现中文关键词检索，无需下载向量模型。",
            metadata={"source": "design.md", "page": 0},
        ),
        Document(
            page_content="用户可以在在线大模型与支持离线的 Ollama 本地模型之间自由切换。",
            metadata={"source": "model.md", "page": 0},
        ),
    ]


class LoaderTests(unittest.TestCase):
    def test_load_utf8_text(self) -> None:
        documents = load_document("笔记.txt", "第一行\n第二行".encode("utf-8"))
        self.assertEqual(len(documents), 1)
        self.assertIn("第二行", documents[0].page_content)
        self.assertEqual(source_label(documents[0]), "笔记.txt")

    def test_load_gbk_text(self) -> None:
        documents = load_document("旧文件.txt", "中文编码测试".encode("gbk"))
        self.assertIn("中文编码测试", documents[0].page_content)

    def test_load_markdown(self) -> None:
        documents = load_document("说明.md", "# 标题\n\n正文内容".encode("utf-8"))
        self.assertIn("正文内容", documents[0].page_content)

    def test_load_docx(self) -> None:
        document = DocxDocument()
        document.add_paragraph("这是 Word 段落")
        table = document.add_table(rows=1, cols=2)
        table.rows[0].cells[0].text = "姓名"
        table.rows[0].cells[1].text = "李诗琪"
        buffer = BytesIO()
        document.save(buffer)

        documents = load_document("作业.docx", buffer.getvalue())
        text = "\n".join(item.page_content for item in documents)
        self.assertIn("这是 Word 段落", text)
        self.assertIn("李诗琪", text)

    def test_unsupported_suffix(self) -> None:
        with self.assertRaises(ValueError):
            load_document("图片.png", b"fake")


class SplitterTests(unittest.TestCase):
    def test_split_keeps_content_and_metadata(self) -> None:
        long_text = "这是一句用于测试的说明。" * 120
        document = Document(
            page_content=long_text, metadata={"source": "long.txt", "page": 0}
        )
        chunks = split_documents([document], chunk_size=200, chunk_overlap=40)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.page_content.strip() for chunk in chunks))
        self.assertTrue(all(chunk.metadata["source"] == "long.txt" for chunk in chunks))

    def test_split_drops_empty_documents(self) -> None:
        chunks = split_documents(
            [Document(page_content="   ", metadata={"source": "empty.txt"})]
        )
        self.assertEqual(chunks, [])


class RetrieverTests(unittest.TestCase):
    def test_bm25_returns_relevant_chunk(self) -> None:
        retriever = build_retriever(sample_documents(), k=1)
        documents = retriever.invoke("jieba 中文检索是怎么做的")
        self.assertEqual(len(documents), 1)
        self.assertIn("BM25", documents[0].page_content)

    def test_bm25_requires_documents(self) -> None:
        with self.assertRaises(ValueError):
            build_retriever([], k=3)

    def test_tokenize_filters_stopwords_and_punctuation(self) -> None:
        tokens = _tokenize("什么是 BM25？它有什么优点。")
        self.assertNotIn("什么", tokens)
        self.assertNotIn("？", tokens)
        self.assertIn("bm25", tokens)
        self.assertIn("优点", tokens)

    def test_retriever_prefers_offline_chunk(self) -> None:
        retriever = build_retriever(sample_documents(), k=1)
        documents = retriever.invoke("支持离线使用吗")
        self.assertIn("Ollama", documents[0].page_content)

    def test_format_docs_contains_number_and_source(self) -> None:
        text = format_docs(sample_documents()[:2])
        self.assertIn("【片段 1｜intro.md】", text)
        self.assertIn("【片段 2｜design.md】", text)

    def test_history_aware_rewrite(self) -> None:
        llm = FakeListChatModel(responses=["LangChain 文档问答 本地模型"])
        retriever = build_retriever(sample_documents(), k=2)
        documents = retrieve_context(
            "它支持离线吗",
            llm,
            retriever,
            [HumanMessage(content="介绍一下这个应用")],
        )
        self.assertGreaterEqual(len(documents), 1)
        self.assertTrue(any("Ollama" in item.page_content for item in documents))


class ChainTests(unittest.TestCase):
    def test_stream_answer(self) -> None:
        llm = FakeListChatModel(responses=["这是基于资料的测试回答。"])
        chunks = list(
            stream_qa_answer("介绍一下", llm, sample_documents()[:1], [])
        )
        self.assertEqual("".join(chunks), "这是基于资料的测试回答。")


if __name__ == "__main__":
    unittest.main()
