"""Agentic RAG graph với multi-layer cache + parallel web crawl.

Flow (PLAN_AGENTIC_RAG_UNIFIED.md §7):
    normalize_query
      → get_embedding
      → check_semantic_cache --hit--> return_cached --> END
      → check_final_cache    --hit--> return_cached --> END
      → route
      → (db_retrieve ‖ parallel_crawl)
      → merge → grade → rewrite|generate
      → store_cache → END
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from vietjet.agents.cache_nodes import (
    after_cache_check,
    check_final_cache_node,
    check_semantic_cache_node,
    get_embedding_node,
    normalize_node,
    return_cached_node,
    store_cache_node,
)
from vietjet.agents.qna_nodes import (
    after_grade,
    db_retrieve_node,
    generate_node,
    grade_node,
    merge_node,
    parallel_crawl_node,
    rewrite_node,
    route_node,
)


class AgenticState(TypedDict, total=False):
    question: str
    query: str
    doc_type: Optional[str]
    boost_tables: bool
    attempts: int
    docs: list[Document]
    web_candidates: list[dict]
    web_chosen_urls: list[str]
    web_docs: list[Document]
    web_skipped_reason: Optional[str]
    merged_docs: list[Document]
    sufficient: bool
    answer: str
    citations: list[str]

    web_search_enabled: bool
    cache_hit: bool
    early_fired: bool
    crawl_session_id: Optional[str]
    background_pages: int

    normalized_query: str
    slots: dict
    intent_realtime: bool
    query_embedding: Optional[list[float]]
    cached_from: Optional[str]
    context_hash: Optional[str]


def _timed_node(name: str, fn):
    async def wrapper(state):
        counter_start = time.perf_counter()
        try:
            return await fn(state)
        finally:
            duration = time.perf_counter() - counter_start
            print(f"[node {name}] duration={duration:.3f}s")

    wrapper.__name__ = name
    return wrapper


def _after_semantic_cache(state: dict) -> str:
    if state.get("cached_from") == "semantic":
        return "return_cached"
    return "check_final_cache"


def _after_final_cache(state: dict) -> str:
    if state.get("cached_from") == "final":
        return "return_cached"
    return "route"


def build_graph(save_image: bool = True):
    g = StateGraph(AgenticState)

    g.add_node("normalize", _timed_node("normalize", normalize_node))
    g.add_node("get_embedding", _timed_node("get_embedding", get_embedding_node))
    g.add_node("check_semantic_cache", _timed_node("check_semantic_cache", check_semantic_cache_node))
    g.add_node("check_final_cache", _timed_node("check_final_cache", check_final_cache_node))
    g.add_node("return_cached", _timed_node("return_cached", return_cached_node))

    g.add_node("route", _timed_node("route", route_node))
    g.add_node("db_retrieve", _timed_node("db_retrieve", db_retrieve_node))
    g.add_node("parallel_crawl", _timed_node("parallel_crawl", parallel_crawl_node))
    g.add_node("merge", _timed_node("merge", merge_node))
    g.add_node("grade", _timed_node("grade", grade_node))
    g.add_node("rewrite", _timed_node("rewrite", rewrite_node))
    g.add_node("generate", _timed_node("generate", generate_node))
    g.add_node("store_cache", _timed_node("store_cache", store_cache_node))

    g.add_edge(START, "normalize")
    g.add_edge("normalize", "get_embedding")
    g.add_edge("get_embedding", "check_semantic_cache")

    g.add_conditional_edges(
        "check_semantic_cache",
        _after_semantic_cache,
        {"return_cached": "return_cached", "check_final_cache": "check_final_cache"},
    )
    g.add_conditional_edges(
        "check_final_cache",
        _after_final_cache,
        {"return_cached": "return_cached", "route": "route"},
    )
    g.add_edge("return_cached", END)

    g.add_edge("route", "db_retrieve")
    g.add_edge("route", "parallel_crawl")
    g.add_edge("db_retrieve", "merge")
    g.add_edge("parallel_crawl", "merge")
    g.add_edge("merge", "grade")
    g.add_conditional_edges(
        "grade",
        after_grade,
        {"generate": "generate", "rewrite": "rewrite"},
    )
    g.add_edge("rewrite", "db_retrieve")
    g.add_edge("rewrite", "parallel_crawl")
    g.add_edge("generate", "store_cache")
    g.add_edge("store_cache", END)

    compiled = g.compile()
    if save_image:
        try:
            img = compiled.get_graph().draw_mermaid_png()
            (Path(__file__).resolve().parent / "graph_agentic.png").write_bytes(img)
        except Exception:
            pass
    return compiled
