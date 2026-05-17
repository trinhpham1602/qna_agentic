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
        # BM25 là sidecar tuỳ chọn. Thiếu file → degrade sang vector-only.
        if BM25_PATH.exists():
            with BM25_PATH.open("rb") as f:
                d = pickle.load(f)
            self.bm25 = d["bm25"]
            self.chunks: list[dict] = d["chunks"]
            self._chunk_by_id = {c["id"]: c for c in self.chunks}
        else:
            print(
                f"[retriever] WARNING: {BM25_PATH} không tồn tại — chạy vector-only. "
                f"Chạy `python -m vietjet.pipeline ingest` để bật hybrid retrieval."
            )
            self.bm25 = None
            self.chunks = []
            self._chunk_by_id = {}

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

    def _vector_ranking(
        self, query: str, k: int, doc_type: str | None
    ) -> list[tuple[Document, float]]:
        filter_dict = {"doc_type": {"$eq": doc_type}} if doc_type else None
        return self.store.similarity_search_with_score(query, k=k, filter=filter_dict)

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        candidates: int = CANDIDATES,
        doc_type: str | None = None,
        boost_tables: bool = False,
    ) -> list[Document]:
        # Vector ranking with optional metadata filter — always available
        vec_ranked = self._vector_ranking(query, candidates, doc_type)
        # Cache Document by id để build kết quả cuối ở vector-only mode
        vec_doc_by_id = {doc.metadata["id"]: doc for doc, _ in vec_ranked}
        vec_ids = [doc.metadata["id"] for doc, _ in vec_ranked]

        if self.bm25 is None:
            # Vector-only path: giữ thứ tự pgvector, tuỳ chọn boost_tables nếu metadata có
            cand_ids = vec_ids[:]
            if boost_tables:
                # ưu tiên các doc có has_table=True trong metadata pgvector
                cand_ids.sort(
                    key=lambda cid: 0 if vec_doc_by_id[cid].metadata.get("has_table") else 1
                )
            cand_ids = cand_ids[:candidates]
        else:
            # Hybrid: BM25 + RRF
            bm25_ids = self._bm25_ranking(query, candidates)
            fused = _rrf([vec_ids, bm25_ids])
            if boost_tables:
                for cid in list(fused):
                    chunk = self._chunk_by_id.get(cid)
                    if chunk and chunk["has_table"]:
                        fused[cid] *= 1.3
            cand_ids = sorted(fused, key=lambda c: -fused[c])[:candidates]

        # Rerank (tuỳ chọn). Chỉ rerank những id có text — ở vector-only mode dùng
        # page_content; ở hybrid dùng chunk text từ sidecar.
        def _text_of(cid: str) -> str:
            if cid in self._chunk_by_id:
                return self._chunk_by_id[cid]["text"][:RERANK_TEXT_CAP]
            doc = vec_doc_by_id.get(cid)
            return (doc.page_content if doc else "")[:RERANK_TEXT_CAP]

        if self.reranker is None or not cand_ids:
            picked = cand_ids[:top_k]
        else:
            pairs = [(query, _text_of(cid)) for cid in cand_ids]
            scores = self.reranker.predict(pairs, batch_size=4, show_progress_bar=False)
            order = sorted(range(len(cand_ids)), key=lambda i: -scores[i])[:top_k]
            picked = [cand_ids[i] for i in order]

        return [self._to_doc(cid, vec_doc_by_id) for cid in picked]

    def _to_doc(self, cid: str, vec_doc_by_id: dict[str, Document] | None = None) -> Document:
        c = self._chunk_by_id.get(cid)
        if c is not None:
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
        if vec_doc_by_id and cid in vec_doc_by_id:
            return vec_doc_by_id[cid]
        # Cuối cùng: fallback empty doc để không crash
        return Document(page_content="", metadata={"id": cid})


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
