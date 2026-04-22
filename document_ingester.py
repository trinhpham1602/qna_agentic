"""
Flexible document ingester supporting multiple source types.

Supported formats:
  - Excel (.xlsx, .xls) — auto-detects FAQ format (situation/eco_standard cols) or generic rows
  - PDF (.pdf)          — via PyMuPDF (falls back to PyPDF)
  - Word (.docx, .doc)  — via docx2txt
  - Plain text (.txt)
  - CSV (.csv)
  - JSON (.json)
  - URL (http/https)    — via WebBaseLoader

Usage:
    ingester = DocumentIngester()
    docs = ingester.load("path/to/file.xlsx", category="Hành lý", risk=1)
    docs = ingester.load("https://example.com/policy")
    # then: add_documents(docs) from rag_workflow
"""

import json
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

_FAQ_COLUMNS = {"situation", "eco_standard", "promo_eco_basic"}


class DocumentIngester:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, source: str, **extra_metadata) -> List[Document]:
        """
        Load documents from a file path or URL.

        Args:
            source: Absolute/relative file path or http(s):// URL.
            **extra_metadata: Key-value pairs attached to every document's metadata.

        Returns:
            List of LangChain Document objects ready for ingestion.
        """
        if source.startswith("http://") or source.startswith("https://"):
            return self._load_url(source, **extra_metadata)

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        dispatch = {
            ".xlsx": self._load_excel,
            ".xls": self._load_excel,
            ".pdf": self._load_pdf,
            ".docx": self._load_docx,
            ".doc": self._load_docx,
            ".txt": self._load_text,
            ".csv": self._load_csv,
            ".json": self._load_json,
        }
        loader = dispatch.get(path.suffix.lower())
        if loader is None:
            supported = list(dispatch)
            raise ValueError(f"Unsupported extension '{path.suffix}'. Supported: {supported}")

        docs = loader(str(path), **extra_metadata)
        logger.info(f"Loaded {len(docs)} documents from '{source}'")
        return docs

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_excel(self, file_path: str, **extra_metadata) -> List[Document]:
        import pandas as pd

        df = pd.read_excel(file_path)
        normalized_cols = set(df.columns.str.strip().str.lower())

        if _FAQ_COLUMNS.issubset(normalized_cols):
            logger.info(f"Detected FAQ format in {file_path}")
            return self._excel_faq(df, file_path, **extra_metadata)

        logger.info(f"Using generic row-per-document for {file_path}")
        return self._excel_generic(df, file_path, **extra_metadata)

    def _excel_faq(self, df, file_path: str, **extra_metadata) -> List[Document]:
        """VietJet-style FAQ Excel: one row = one policy situation."""
        docs = []
        for _, row in df.iterrows():
            def g(col):
                return str(row.get(col, "") or "")

            content = (
                f"Tình huống: {g('situation')}\n"
                f"Quy định Eco: {g('eco_standard')}\n"
                f"Quy định Promo: {g('promo_eco_basic')}\n"
                f"Quy định Deluxe: {g('deluxe')}\n"
                f"Quy định SkyBoss: {g('skyboss')}\n"
                f"Ghi chú: {g('important_notes')}\n"
                f"Trả lời mẫu: {g('short_answer')}"
            )
            raw_risk = row.get("risk", 1)
            metadata = {
                "source": file_path,
                "source_type": "excel_faq",
                "category": g("category"),
                "risk": int(raw_risk) if str(raw_risk).isdigit() else 1,
                "escalate_when": g("escalate_when"),
            }
            metadata.update(extra_metadata)
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def _excel_generic(self, df, file_path: str, **extra_metadata) -> List[Document]:
        import pandas as pd

        docs = []
        for idx, row in df.iterrows():
            content = "\n".join(
                f"{col}: {val}" for col, val in row.items() if pd.notna(val)
            )
            metadata = {"source": file_path, "source_type": "excel", "row": int(idx)}
            metadata.update(extra_metadata)
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def _load_pdf(self, file_path: str, **extra_metadata) -> List[Document]:
        try:
            from langchain_community.document_loaders import PyMuPDFLoader
            loader = PyMuPDFLoader(file_path)
        except ImportError:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)

        docs = self.splitter.split_documents(loader.load())
        for doc in docs:
            doc.metadata.update({"source": file_path, "source_type": "pdf", **extra_metadata})
        return docs

    def _load_docx(self, file_path: str, **extra_metadata) -> List[Document]:
        from langchain_community.document_loaders import Docx2txtLoader

        docs = self.splitter.split_documents(Docx2txtLoader(file_path).load())
        for doc in docs:
            doc.metadata.update({"source": file_path, "source_type": "docx", **extra_metadata})
        return docs

    def _load_text(self, file_path: str, **extra_metadata) -> List[Document]:
        from langchain_community.document_loaders import TextLoader

        docs = self.splitter.split_documents(
            TextLoader(file_path, encoding="utf-8").load()
        )
        for doc in docs:
            doc.metadata.update({"source": file_path, "source_type": "text", **extra_metadata})
        return docs

    def _load_csv(self, file_path: str, **extra_metadata) -> List[Document]:
        from langchain_community.document_loaders.csv_loader import CSVLoader

        docs = CSVLoader(file_path).load()
        for doc in docs:
            doc.metadata.update({"source": file_path, "source_type": "csv", **extra_metadata})
        return docs

    def _load_json(self, file_path: str, **extra_metadata) -> List[Document]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = data if isinstance(data, list) else [data]
        docs = []
        for idx, item in enumerate(items):
            content = json.dumps(item, ensure_ascii=False, indent=2)
            metadata = {"source": file_path, "source_type": "json", "index": idx}
            metadata.update(extra_metadata)
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def _load_url(self, url: str, **extra_metadata) -> List[Document]:
        from langchain_community.document_loaders import WebBaseLoader

        docs = self.splitter.split_documents(WebBaseLoader(url).load())
        for doc in docs:
            doc.metadata.update({"source": url, "source_type": "url", **extra_metadata})
        return docs
