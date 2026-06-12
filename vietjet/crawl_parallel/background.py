"""BackgroundIngest: lưu pages crawl được vào pgvector.

Pipeline: PageItem → clean markdown → chunk → embed → upsert pgvector
với metadata { last_crawled_at, source_query, crawl_session_id, doc_type=web_live }.

Khác với `vietjet.ingest.main()` (rebuild full collection), background ingest
chỉ ADD documents — không xoá collection. ID dùng `{url_hash}::{idx}` để
idempotent (cùng URL ingest lại sẽ ghi đè).
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from datetime import datetime, timezone
from typing import Iterable

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

from vietjet.chunk import pack_paragraphs, split_prose_and_tables, split_sections
from vietjet.config import (
    COLLECTION_NAME,
    DB_CONNECTION_STRING,
    EMBED_MODEL,
    PARALLEL_INGEST_BATCH_SIZE,
)
from vietjet.crawl_parallel.agent import PageItem
from vietjet.crawl_parallel.text_clean import clean_ingest_text


_MAX_CHARS = 1800
_OVERLAP = 200


def _url_hash(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def _chunk_page(page: PageItem) -> Iterable[dict]:
    md = clean_ingest_text(page.markdown)
    if not md:
        return
    uhash = _url_hash(page.url)
    idx = 0
    for section_path, body in split_sections(md):
        if not body.strip():
            continue
        for segment, is_table in split_prose_and_tables(body):
            if is_table:
                preamble = f"[Mục: {section_path}]\n\n" if section_path else ""
                yield {
                    "id": f"{uhash}::{idx:03d}",
                    "text": f"{preamble}{segment}".strip(),
                    "section_path": section_path or "web",
                    "has_table": True,
                }
                idx += 1
            else:
                for piece in pack_paragraphs(segment, _MAX_CHARS, _OVERLAP):
                    yield {
                        "id": f"{uhash}::{idx:03d}",
                        "text": piece,
                        "section_path": section_path or "web",
                        "has_table": False,
                    }
                    idx += 1
    if idx == 0:
        # Không tách được section nào → emit doc thô (cap)
        yield {
            "id": f"{uhash}::000",
            "text": md[:_MAX_CHARS],
            "section_path": "web",
            "has_table": False,
        }


class BackgroundIngest:
    def __init__(
        self,
        ingest_queue: asyncio.Queue,
        session_id: str,
        source_query: str,
        *,
        batch_size: int = PARALLEL_INGEST_BATCH_SIZE,
        idle_timeout: float = 5.0,
    ) -> None:
        self.ingest_queue = ingest_queue
        self.session_id = session_id
        self.source_query = source_query
        self.batch_size = batch_size
        self.idle_timeout = idle_timeout
        self.ingested_pages: int = 0
        self.ingested_chunks: int = 0
        self._store: PGVector | None = None
        self._embedder: HuggingFaceEmbeddings | None = None

    def _get_store(self) -> PGVector:
        if self._store is None:
            self._embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
            self._store = PGVector(
                embeddings=self._embedder,
                connection=DB_CONNECTION_STRING,
                collection_name=COLLECTION_NAME,
                use_jsonb=True,
                pre_delete_collection=False,
            )
        return self._store

    async def run(self) -> None:
        from vietjet.crawl_parallel.agent import SENTINEL

        batch: list[PageItem] = []
        while True:
            try:
                item = await asyncio.wait_for(
                    self.ingest_queue.get(), timeout=self.idle_timeout
                )
            except asyncio.TimeoutError:
                if batch:
                    await self._flush(batch)
                    batch = []
                continue

            if item is SENTINEL:
                if batch:
                    await self._flush(batch)
                print(
                    f"[bg-ingest] done. pages={self.ingested_pages} chunks={self.ingested_chunks}"
                )
                return
            if not isinstance(item, PageItem):
                continue
            batch.append(item)
            if len(batch) >= self.batch_size:
                await self._flush(batch)
                batch = []

    async def _flush(self, pages: list[PageItem]) -> None:
        if not pages:
            return
        try:
            await asyncio.to_thread(self._flush_sync, pages)
        except Exception as exc:
            print(f"[bg-ingest] flush failed: {exc}")

    def _flush_sync(self, pages: list[PageItem]) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        now_ts = time.time()
        store = self._get_store()
        docs: list[Document] = []
        ids: list[str] = []
        for page in pages:
            page_chunks = list(_chunk_page(page))
            if not page_chunks:
                continue
            for c in page_chunks:
                docs.append(Document(
                    page_content=c["text"],
                    metadata={
                        "id": c["id"],
                        "source": page.url,
                        "doc_type": "web_live",
                        "section_path": c["section_path"],
                        "has_table": c["has_table"],
                        "title": page.title,
                        "last_crawled_at": now_iso,
                        "last_crawled_ts": now_ts,
                        "source_query": self.source_query,
                        "crawl_session_id": self.session_id,
                    },
                ))
                ids.append(c["id"])
            self.ingested_pages += 1
            self.ingested_chunks += len(page_chunks)
        if not docs:
            return
        # langchain_postgres upsert: nếu id trùng thì add_documents sẽ ghi đè
        store.add_documents(docs, ids=ids)
        print(
            f"[bg-ingest] flushed batch: {len(pages)} pages → {len(docs)} chunks "
            f"(total pages={self.ingested_pages})"
        )
