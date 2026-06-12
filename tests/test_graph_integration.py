"""Integration test: combined_agent.build_graph() chạy với parallel_crawl_node.

Mock toàn bộ I/O:
  - CrawlCoordinator.stream → yield giả cache_hit + done
  - db_retrieve → trả [] (không cần pgvector)
  - generate_agent → echo answer
  - classify_intent LLM → trả "question"
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.documents import Document


def _fake_event(type_: str, **payload):
    return SimpleNamespace(type=type_, payload=payload)


async def _fake_stream(self, query):
    """Fake CrawlCoordinator.stream() yielding cache_hit + done."""
    yield _fake_event(
        "cache_hit",
        docs=[{
            "source": "https://vietjetair.com/baggage",
            "section_path": "Hành lý",
            "doc_type": "web_live",
            "title": "Phí hành lý",
            "last_crawled_at": "2026-05-24T10:00:00+00:00",
            "content": "Phí ký gửi 20kg là 200.000 VND. Áp dụng nội địa.",
        }],
        count=1,
    )
    yield _fake_event("done", reason="cache_hit")


async def _fake_db_retrieve(query, doc_type=None, boost_tables=False, top_k=4, candidates=20):
    return []  # Không có DB seed


async def _fake_generate(question, db_docs, web_docs):
    snippets = []
    for d in web_docs:
        snippets.append(d.page_content[:80])
    return (
        f"[fake-answer] question={question[:40]} | web_docs={len(web_docs)} | first={snippets[:1]}",
        [f"web:{d.metadata.get('source')}" for d in web_docs],
    )


async def _fake_classify_llm(self, messages):
    """Mock ChatOllama.ainvoke for classify_intent → return JSON intent=question."""
    return SimpleNamespace(content='{"intent": "question"}')


def test_full_graph_question_with_cache_hit():
    """End-to-end: classify_intent → qna_route → (db_retrieve || parallel_crawl) → merge → grade → generate."""

    async def _run():
        from vietjet.crawl_parallel.coordinator import CrawlCoordinator

        with patch.object(CrawlCoordinator, "stream", _fake_stream), \
             patch("vietjet.qna_nodes.db_retrieve", _fake_db_retrieve), \
             patch("vietjet.qna_nodes.generate_agent", _fake_generate), \
             patch("vietjet.qna_nodes.grade_agent", AsyncMock(return_value=True)), \
             patch("vietjet.combined_agent._llm") as mock_llm_factory:

            mock_llm = SimpleNamespace(ainvoke=_fake_classify_llm.__get__(SimpleNamespace()))
            mock_llm_factory.return_value = mock_llm

            from vietjet.combined_agent import build_graph

            graph = build_graph(save_image=False)
            state = {
                "slots": {},
                "attempts": 0,
                "done": False,
                "user_input": "phí hành lý 20kg",
                "question": "",
                "intent": None,
                "answer": "",
                "slot_question": "",
                "web_candidates": [],
                "web_chosen_urls": [],
                "web_docs": [],
                "web_skipped_reason": None,
                "merged_docs": [],
                "cache_hit": False,
                "early_fired": False,
                "crawl_session_id": None,
                "background_pages": 0,
            }
            out = await graph.ainvoke(state)

        # Verify
        assert out["intent"] == "question", out.get("intent")
        assert out["cache_hit"] is True, "should propagate cache_hit"
        assert len(out["web_docs"]) >= 1, "should have 1 web doc from cache"
        assert "vietjetair.com/baggage" in (out["web_chosen_urls"] or [""])[0]
        assert out["done"] is True
        assert "fake-answer" in out["answer"]
        print("answer:", out["answer"])
        print("citations:", out["citations"])
        print("cache_hit:", out["cache_hit"])
        print("web_docs:", len(out["web_docs"]))

    asyncio.run(_run())


def test_parallel_crawl_node_directly():
    """Test riêng node parallel_crawl với mock coordinator."""

    async def _run():
        from vietjet.crawl_parallel.coordinator import CrawlCoordinator
        from vietjet.qna_nodes import parallel_crawl_node

        with patch.object(CrawlCoordinator, "stream", _fake_stream):
            out = await parallel_crawl_node({"query": "phí hành lý", "question": "phí hành lý"})

        assert out["cache_hit"] is True
        assert len(out["web_docs"]) == 1
        assert out["web_skipped_reason"] is None
        assert isinstance(out["web_docs"][0], Document)
        print("OK parallel_crawl_node direct test")

    asyncio.run(_run())


def test_parallel_crawl_node_empty_query():
    async def _run():
        from vietjet.qna_nodes import parallel_crawl_node
        out = await parallel_crawl_node({"query": "", "question": ""})
        assert out["web_docs"] == []
        assert out["web_skipped_reason"] == "empty_query"

    asyncio.run(_run())


def test_parallel_crawl_node_with_partial_answer():
    """Verify node parses partial_answer event correctly."""

    async def _fake_stream_partial(self, query):
        yield _fake_event(
            "partial_answer",
            results=[
                {
                    "url": "https://x.com/a",
                    "snippet": "answer snippet for query",
                    "sim": 0.85,
                    "confidence": "high",
                    "title": "Page A",
                }
            ],
            session_id="sess-123",
            reason="early_match",
            early_fired=True,
        )
        yield _fake_event("ingested", pages=2, chunks=8)
        yield _fake_event("done", session_id="sess-123")

    async def _run():
        from vietjet.crawl_parallel.coordinator import CrawlCoordinator
        from vietjet.qna_nodes import parallel_crawl_node

        with patch.object(CrawlCoordinator, "stream", _fake_stream_partial):
            out = await parallel_crawl_node({"query": "test", "question": "test"})

        assert out["early_fired"] is True
        assert out["cache_hit"] is False
        assert out["crawl_session_id"] == "sess-123"
        assert out["background_pages"] == 2
        assert len(out["web_docs"]) == 1
        assert out["web_docs"][0].metadata["source"] == "https://x.com/a"
        print("OK partial_answer parsing")

    asyncio.run(_run())


# ─── qna_agentic standalone graph (no intent classify) ──────────────────


async def _miss_semantic(state):
    return {"cached_from": None}


async def _miss_final(state):
    return {"cached_from": state.get("cached_from")}


async def _stub_embedding(state):
    return {"query_embedding": [0.0] * 768}


def test_qna_agentic_graph_end_to_end():
    """qna_agentic.build_graph: route → (db_retrieve ‖ parallel_crawl) → merge → grade → generate."""

    async def _run():
        from vietjet.crawl_parallel.coordinator import CrawlCoordinator

        with patch.object(CrawlCoordinator, "stream", _fake_stream), \
             patch("vietjet.qna_nodes.db_retrieve", _fake_db_retrieve), \
             patch("vietjet.qna_nodes.generate_agent", _fake_generate), \
             patch("vietjet.qna_nodes.grade_agent", AsyncMock(return_value=True)), \
             patch("vietjet.qna_graph.check_semantic_cache_node", _miss_semantic), \
             patch("vietjet.qna_graph.check_final_cache_node", _miss_final), \
             patch("vietjet.qna_graph.get_embedding_node", _stub_embedding):

            from vietjet.qna_agentic import _initial_state
            from vietjet.qna_graph import build_graph

            graph = build_graph(save_image=False)
            out = await graph.ainvoke(_initial_state("phí hành lý quốc nội"))

        assert out["cache_hit"] is True
        assert len(out["web_docs"]) >= 1
        assert "fake-answer" in out["answer"]
        assert out["sufficient"] is True
        assert out["attempts"] == 0
        print("qna_agentic answer:", out["answer"][:80])
        print("qna_agentic doc_type:", out.get("doc_type"))

    asyncio.run(_run())


def test_qna_agentic_rewrite_loop():
    """Khi grade=False lần đầu, graph rewrite rồi loop lại retrieve+crawl, lần 2 grade=True."""

    async def _run():
        from vietjet.crawl_parallel.coordinator import CrawlCoordinator

        # grade trả False lần đầu, True lần sau
        calls = {"n": 0}

        async def alt_grade(question, docs):
            calls["n"] += 1
            return calls["n"] >= 2

        async def alt_rewrite(question):
            return question + " (rewritten)"

        with patch.object(CrawlCoordinator, "stream", _fake_stream), \
             patch("vietjet.qna_nodes.db_retrieve", _fake_db_retrieve), \
             patch("vietjet.qna_nodes.generate_agent", _fake_generate), \
             patch("vietjet.qna_nodes.grade_agent", alt_grade), \
             patch("vietjet.qna_nodes.rewrite_agent", alt_rewrite), \
             patch("vietjet.qna_graph.check_semantic_cache_node", _miss_semantic), \
             patch("vietjet.qna_graph.check_final_cache_node", _miss_final), \
             patch("vietjet.qna_graph.get_embedding_node", _stub_embedding):

            from vietjet.qna_agentic import _initial_state
            from vietjet.qna_graph import build_graph

            graph = build_graph(save_image=False)
            out = await graph.ainvoke(_initial_state("test loop"))

        assert calls["n"] == 2, f"grade should be called 2x, got {calls['n']}"
        assert out["attempts"] == 1, f"should rewrite once, got attempts={out.get('attempts')}"
        assert "rewritten" in out["query"]
        print("rewrite loop OK, final query:", out["query"])

    asyncio.run(_run())


def test_qna_agentic_with_partial_answer_early_fire():
    """Verify early_fired propagates qua qna_agentic graph."""

    async def _fake_stream_partial(self, query):
        yield _fake_event(
            "partial_answer",
            results=[{
                "url": "https://vietjetair.com/policy",
                "snippet": "Phí đổi vé Eco quốc nội 360.000 VND",
                "sim": 0.92,
                "confidence": "high",
                "title": "Phí đổi vé",
            }],
            session_id="sess-abc",
            reason="early_match",
            early_fired=True,
        )
        yield _fake_event("ingested", pages=3, chunks=12)
        yield _fake_event("done", session_id="sess-abc")

    async def _run():
        from vietjet.crawl_parallel.coordinator import CrawlCoordinator

        with patch.object(CrawlCoordinator, "stream", _fake_stream_partial), \
             patch("vietjet.qna_nodes.db_retrieve", _fake_db_retrieve), \
             patch("vietjet.qna_nodes.generate_agent", _fake_generate), \
             patch("vietjet.qna_nodes.grade_agent", AsyncMock(return_value=True)), \
             patch("vietjet.qna_graph.check_semantic_cache_node", _miss_semantic), \
             patch("vietjet.qna_graph.check_final_cache_node", _miss_final), \
             patch("vietjet.qna_graph.get_embedding_node", _stub_embedding):

            from vietjet.qna_agentic import _initial_state
            from vietjet.qna_graph import build_graph

            graph = build_graph(save_image=False)
            out = await graph.ainvoke(_initial_state("phí đổi vé eco"))

        assert out["cache_hit"] is False
        assert out["early_fired"] is True
        assert out["crawl_session_id"] == "sess-abc"
        assert out["background_pages"] == 3
        assert len(out["web_docs"]) == 1
        print("early_fired propagated. session:", out["crawl_session_id"])

    asyncio.run(_run())


if __name__ == "__main__":
    import inspect
    import sys

    failures = 0
    mod = sys.modules[__name__]
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"OK   {name}")
        except Exception as exc:
            import traceback
            print(f"FAIL {name}")
            traceback.print_exc()
            failures += 1
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
