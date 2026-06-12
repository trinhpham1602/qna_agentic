"""Verify the live-ingest flow: empty DB + parallel_crawl provides docs.

Khi user clear hết data và muốn DB ingest qua câu hỏi:
  - db_retrieve_node return [] (DB rỗng) → graph KHÔNG crash
  - parallel_crawl_node trả web_docs → merge có docs
  - grade → generate → store_cache
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


def _fake_event(type_: str, **payload):
    return SimpleNamespace(type=type_, payload=payload)


async def _stream_partial_answer(self, query):
    yield _fake_event(
        "partial_answer",
        results=[
            {
                "url": "https://vietjetair.com/test-policy",
                "snippet": "Phí đổi vé Eco quốc nội 360.000 VND",
                "sim": 0.91,
                "confidence": "high",
                "title": "Phí đổi vé",
            }
        ],
        session_id="live-1",
        reason="early_match",
        early_fired=True,
    )
    yield _fake_event("ingested", pages=1, chunks=4)
    yield _fake_event("done", session_id="live-1")


async def _empty_db_retrieve(query, doc_type=None, boost_tables=False, top_k=4, candidates=20):
    return []


async def _fake_generate(question, db_docs, web_docs):
    return (
        f"[live-answer] q={question[:30]} web={len(web_docs)} db={len(db_docs)}",
        [f"web:{d.metadata.get('source')}" for d in web_docs],
    )


async def _miss_semantic(state):
    return {"cached_from": None}


async def _miss_final(state):
    return {"cached_from": state.get("cached_from")}


async def _stub_embedding(state):
    return {"query_embedding": [0.01] * 768}


async def _noop_store_cache(state):
    return {}


def test_empty_db_falls_back_to_web_crawl():
    async def _run():
        from vietjet.crawl_parallel.coordinator import CrawlCoordinator

        with patch.object(CrawlCoordinator, "stream", _stream_partial_answer), \
             patch("vietjet.qna_nodes.db_retrieve", _empty_db_retrieve), \
             patch("vietjet.qna_nodes.generate_agent", _fake_generate), \
             patch("vietjet.qna_nodes.grade_agent", AsyncMock(return_value=True)), \
             patch("vietjet.qna_graph.check_semantic_cache_node", _miss_semantic), \
             patch("vietjet.qna_graph.check_final_cache_node", _miss_final), \
             patch("vietjet.qna_graph.get_embedding_node", _stub_embedding), \
             patch("vietjet.qna_graph.store_cache_node", _noop_store_cache):

            from vietjet.qna_agentic import _initial_state
            from vietjet.qna_graph import build_graph

            graph = build_graph(save_image=False)
            out = await graph.ainvoke(_initial_state("phí đổi vé eco bao nhiêu"))

        assert out["docs"] == []
        assert len(out["web_docs"]) == 1
        assert out["early_fired"] is True
        assert "live-answer" in out["answer"]
        assert "web=1 db=0" in out["answer"]

    asyncio.run(_run())


def test_retriever_handles_missing_reranker_gracefully():
    from vietjet.retriever import get_retriever

    r = get_retriever(use_rerank=True)
    docs = r.search("any query", top_k=4)
    assert isinstance(docs, list)
