"""文档解析：把上传的 PDF、DOCX、TXT、Markdown 转成 LangChain Document。"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from langchain_core.documents import Document

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".pdf", ".docx"}
_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")


def _decode(data: bytes) -> str:
    for encoding in _ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _load_text(file_name: str, data: bytes) -> list[Document]:
    text = _decode(data)
    if not text.strip():
        return []
    return [Document(page_content=text, metadata={"source": file_name, "page": 0})]


def _load_pdf(file_name: str, data: bytes) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    documents: list[Document] = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": file_name, "page": index},
                )
            )
    return documents


def _load_docx(file_name: str, data: bytes) -> list[Document]:
    from docx import Document as DocxDocument

    document = DocxDocument(BytesIO(data))
    blocks: list[str] = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            blocks.append(text)

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            line = " | ".join(cell for cell in cells if cell)
            if line:
                blocks.append(line)

    text = "\n".join(blocks)
    if not text.strip():
        return []
    return [Document(page_content=text, metadata={"source": file_name, "page": 0})]


def load_document(file_name: str, data: bytes) -> list[Document]:
    """按扩展名解析单个文件，返回若干 LangChain Document。"""
    suffix = Path(file_name).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return _load_text(file_name, data)
    if suffix == ".pdf":
        return _load_pdf(file_name, data)
    if suffix == ".docx":
        return _load_docx(file_name, data)
    raise ValueError(
        f"暂不支持 {suffix or '无扩展名'} 格式，请上传 PDF、DOCX、TXT 或 Markdown 文件。"
    )


def source_label(document: Document) -> str:
    """生成用于界面展示的来源标签。"""
    source = Path(str(document.metadata.get("source", "未知文件"))).name
    page = document.metadata.get("page")
    if isinstance(page, int) and page > 0:
        return f"{source} · 第 {page} 页"
    return source
