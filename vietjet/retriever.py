"""Hybrid retrieval: pgvector + BM25 (RRF) → optional cross-encoder rerank.

The retriever is a singleton so a long-lived process (FastAPI / CLI) loads
the embedder, reranker and BM25 once.
"""

from __future__ import annotations
import pickle
from functools import lru_cache
from typing import Iterable

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from sentence_transformers import CrossEncoder

from vietjet.config import (
    BM25_PATH,
    CANDIDATES,
    COLLECTION_NAME,
    DB_CONNECTION_STRING,
    EMBED_MODEL,
    RERANK_MAX_LENGTH,
    RERANK_MODEL,
    RERANK_TEXT_CAP,
    RRF_K,
    TOP_K,
)
from vietjet.ingest import tokenize_vn


def _rrf(rankings: Iterable[list[str]]) -> dict[str, float]:
    """Reciprocal rank fusion keyed by chunk id."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            score[cid] = score.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
    return score


class VietjetRetriever:
    def __init__(self, use_rerank: bool = True):
        with BM25_PATH.open("rb") as f:
            d = pickle.load(f)
        self.bm25 = d["bm25"]
        self.chunks: list[dict] = d["chunks"]
        self._chunk_by_id = {c["id"]: c for c in self.chunks}

        self.embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.store = PGVector(
            embeddings=self.embedder,
            connection=DB_CONNECTION_STRING,
            collection_name=COLLECTION_NAME,
            use_jsonb=True,
        )
        self.reranker = (
            CrossEncoder(RERANK_MODEL, max_length=RERANK_MAX_LENGTH) if use_rerank else None
        )

    def _bm25_ranking(self, query: str, k: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize_vn(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self.chunks[i]["id"] for i in order]

    def _vector_ranking(self, query: str, k: int, doc_type: str | None) -> list[tuple[str, float]]:
        filter_dict = {"doc_type": {"$eq": doc_type}} if doc_type else None
        results = self.store.similarity_search_with_score(query, k=k, filter=filter_dict)
        return [(doc.metadata["id"], score) for doc, score in results]

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        candidates: int = CANDIDATES,
        doc_type: str | None = None,
        boost_tables: bool = False,
    ) -> list[Document]:
        # Vector ranking with optional metadata filter
        vec_ranked = self._vector_ranking(query, candidates, doc_type)
        vec_ids = [cid for cid, _ in vec_ranked]
        # BM25 always over the whole corpus (filtering BM25 by metadata is awkward;
        # we let RRF demote out-of-type chunks via the vector side instead).
        bm25_ids = self._bm25_ranking(query, candidates)

        fused = _rrf([vec_ids, bm25_ids])

        if boost_tables:
            for cid in list(fused):
                if self._chunk_by_id[cid]["has_table"]:
                    fused[cid] *= 1.3

        cand_ids = sorted(fused, key=lambda c: -fused[c])[:candidates]

        if self.reranker is None:
            picked = cand_ids[:top_k]
        else:
            pairs = [(query, self._chunk_by_id[cid]["text"][:RERANK_TEXT_CAP]) for cid in cand_ids]
            scores = self.reranker.predict(pairs, batch_size=4, show_progress_bar=False)
            order = sorted(range(len(cand_ids)), key=lambda i: -scores[i])[:top_k]
            picked = [cand_ids[i] for i in order]

        return [self._to_doc(cid) for cid in picked]

    def _to_doc(self, cid: str) -> Document:
        c = self._chunk_by_id[cid]
        return Document(
            page_content=c["text"],
            metadata={
                "id": c["id"],
                "source": c["source"],
                "doc_type": c["doc_type"],
                "section_path": c["section_path"],
                "has_table": c["has_table"],
            },
        )


@lru_cache(maxsize=1)
def get_retriever(use_rerank: bool = True) -> VietjetRetriever:
    return VietjetRetriever(use_rerank=use_rerank)


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "phí ký gửi 20kg quốc nội"
    r = get_retriever()
    for i, d in enumerate(r.search(q, top_k=4), 1):
        m = d.metadata
        print(f"\n[{i}] {m['source']} | {m['section_path']} | type={m['doc_type']} | table={m['has_table']}")
        print(d.page_content[:240].replace("\n", " ") + "…")
