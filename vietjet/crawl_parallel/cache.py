"""CacheChecker: check pgvector cho doc đã crawl gần đây + đủ liên quan.

Logic:
  1. Query pgvector với filter doc_type=web_live AND last_crawled_ts >= now - TTL
     (filter JSONB qua langchain_postgres similarity_search_with_score)
  2. Nếu top-K result có score >= threshold → cache hit
     → return list[Document]
  3. Ngược lại → cache miss → crawl mới

Khác với retrieve nội bộ (`retriever.search`):
  - Cache check chỉ xem doc_type=web_live (đã crawl từ web, không phải seed data)
  - Có TTL filter (1h mặc định)
  - Trả về list Document raw, không rerank
"""

from __future__ import annotations

import time

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

from vietjet.config import (
    COLLECTION_NAME,
    DB_CONNECTION_STRING,
    EMBED_MODEL,
    PARALLEL_CACHE_SIM_THRESHOLD,
    PARALLEL_CACHE_TTL_SECONDS,
)


class CacheChecker:
    def __init__(
        self,
        *,
        ttl_seconds: float = PARALLEL_CACHE_TTL_SECONDS,
        sim_threshold: float = PARALLEL_CACHE_SIM_THRESHOLD,
        top_k: int = 5,
    ) -> None:
        self.ttl = ttl_seconds
        self.sim_threshold = sim_threshold
        self.top_k = top_k
        self._store: PGVector | None = None

    def _get_store(self) -> PGVector:
        if self._store is None:
            embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
            self._store = PGVector(
                embeddings=embedder,
                connection=DB_CONNECTION_STRING,
                collection_name=COLLECTION_NAME,
                use_jsonb=True,
            )
        return self._store

    def check(self, query: str) -> tuple[bool, list[Document]]:
        """Trả (hit, docs). hit=True nếu top-K có doc web_live còn hạn và đủ sim.

        Hàm này SYNC để dễ gọi qua asyncio.to_thread từ coordinator —
        similarity_search_with_score của langchain_postgres vốn là sync.
        """
        cutoff_ts = time.time() - self.ttl
        flt = {
            "$and": [
                {"doc_type": {"$eq": "web_live"}},
                {"last_crawled_ts": {"$gte": cutoff_ts}},
            ]
        }
        try:
            results = self._get_store().similarity_search_with_score(
                query, k=self.top_k, filter=flt
            )
        except Exception as exc:
            print(f"[cache-check] query failed: {exc}")
            return False, []

        if not results:
            print(f"[cache-check] no web_live docs within TTL ({self.ttl}s)")
            return False, []

        # PGVector trả về distance (lower = closer). Convert sang sim ~ 1 - dist.
        scored: list[tuple[Document, float]] = []
        for doc, dist in results:
            sim = max(0.0, 1.0 - float(dist))
            scored.append((doc, sim))
        best_sim = scored[0][1]
        print(
            f"[cache-check] best_sim={best_sim:.3f} (threshold={self.sim_threshold}) "
            f"top={scored[0][0].metadata.get('source')}"
        )
        if best_sim < self.sim_threshold:
            return False, []
        return True, [d for d, _ in scored]
