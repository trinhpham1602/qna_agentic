from __future__ import annotations

import pytest

from vietjet.qna_graph import build_graph


def test_graph_has_cache_nodes():
    g = build_graph(save_image=False)
    nodes = set(g.get_graph().nodes)
    expected = {
        "normalize",
        "get_embedding",
        "check_semantic_cache",
        "check_final_cache",
        "return_cached",
        "route",
        "db_retrieve",
        "parallel_crawl",
        "merge",
        "grade",
        "rewrite",
        "generate",
        "store_cache",
    }
    missing = expected - nodes
    assert not missing, f"missing nodes: {missing}"


def test_graph_entry_point_is_normalize():
    g = build_graph(save_image=False)
    edges = list(g.get_graph().edges)
    from_start = [e for e in edges if e.source == "__start__"]
    assert len(from_start) == 1
    assert from_start[0].target == "normalize"


def test_graph_store_cache_before_end():
    g = build_graph(save_image=False)
    edges = list(g.get_graph().edges)
    to_end_sources = [e.source for e in edges if e.target == "__end__"]
    assert "store_cache" in to_end_sources
    assert "return_cached" in to_end_sources
