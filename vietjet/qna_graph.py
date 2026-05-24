from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from vietjet.qna_nodes import (
    after_grade,
    db_retrieve_node,
    generate_node,
    grade_node,
    link_judge_node,
    merge_node,
    rewrite_node,
    route_node,
    web_fetch_node,
    web_search_node,
)


class AgenticState(TypedDict, total=False):
    question: str
    query: str
    doc_type: str | None
    boost_tables: bool
    attempts: int
    docs: list[Document]
    web_candidates: list[dict]
    web_chosen_urls: list[str]
    web_docs: list[Document]
    web_skipped_reason: str | None
    merged_docs: list[Document]
    sufficient: bool
    answer: str
    citations: list[str]


def build_graph(save_image: bool = True):
    g = StateGraph(AgenticState)

    g.add_node("route", route_node)
    g.add_node("db_retrieve", db_retrieve_node)
    g.add_node("web_search", web_search_node)
    g.add_node("link_judge", link_judge_node)
    g.add_node("web_fetch", web_fetch_node)
    g.add_node("merge", merge_node)
    g.add_node("grade", grade_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("generate", generate_node)

    g.add_edge(START, "route")
    g.add_edge("route", "db_retrieve")
    g.add_edge("route", "web_search")
    g.add_edge("web_search", "link_judge")
    g.add_edge("link_judge", "web_fetch")
    g.add_edge("db_retrieve", "merge")
    g.add_edge("web_fetch", "merge")
    g.add_edge("merge", "grade")
    g.add_conditional_edges(
        "grade",
        after_grade,
        {"generate": "generate", "rewrite": "rewrite"},
    )
    g.add_edge("rewrite", "db_retrieve")
    g.add_edge("rewrite", "web_search")
    g.add_edge("generate", END)

    compiled = g.compile()
    if save_image:
        try:
            img = compiled.get_graph().draw_mermaid_png()
            (Path(__file__).resolve().parent / "graph_agentic.png").write_bytes(img)
        except Exception:
            pass
    return compiled
