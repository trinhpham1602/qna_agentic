from __future__ import annotations

import asyncio

from langchain_core.documents import Document

from vietjet.config import CANDIDATES, TOP_K
from vietjet.retrieval.retriever import get_retriever


async def db_retrieve(
    query: str,
    doc_type: str | None = None,
    boost_tables: bool = False,
    top_k: int = TOP_K,
    candidates: int = CANDIDATES,
) -> list[Document]:
    try:
        retriever = get_retriever(use_rerank=True)
    except Exception as exc:
        print(f"[db_retrieve] retriever init failed: {exc} — skipping DB branch.")
        return []
    try:
        return await asyncio.to_thread(
            retriever.search,
            query,
            top_k,
            candidates,
            doc_type,
            boost_tables,
        )
    except Exception as exc:
        print(f"[db_retrieve] search failed: {exc} — returning empty.")
        return []
