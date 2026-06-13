"""Standalone QnA entry-point (no intent classification, no slot-filling).

Dùng cho:
  - CLI: `python -m vietjet.agents.qna_agentic "câu hỏi"`
  - HTTP endpoint `/qna` (sync) hoặc `/qna-stream` (SSE) trong vietjet.app.server

Graph từ vietjet.agents.qna_graph.build_graph() có fan-out:
    route → (db_retrieve ‖ parallel_crawl) → merge → grade → rewrite|generate
"""

from __future__ import annotations

import asyncio
from typing import Any

from vietjet.agents.qna_graph import AgenticState, build_graph
from vietjet.config import WEB_SEARCH_ENABLED

_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def _initial_state(question: str, web_search: bool = WEB_SEARCH_ENABLED) -> AgenticState:
    return {
        "question": question,
        "web_search_enabled": web_search,
        "attempts": 0,
        "docs": [],
        "web_candidates": [],
        "web_chosen_urls": [],
        "web_docs": [],
        "web_skipped_reason": None,
        "merged_docs": [],
        "cache_hit": False,
        "early_fired": False,
        "crawl_session_id": None,
        "background_pages": 0,
        "normalized_query": "",
        "slots": {},
        "intent_realtime": False,
        "query_embedding": None,
        "cached_from": None,
        "context_hash": None,
    }


async def ask(question: str, web_search: bool = WEB_SEARCH_ENABLED) -> dict[str, Any]:
    graph = get_graph()
    return await graph.ainvoke(_initial_state(question, web_search))


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Phí đổi vé Vietjet hạng eco quốc nội bao nhiêu"
    out = asyncio.run(ask(q))

    print("\n=== Question ===")
    print(q)
    print(
        f"\n=== Doc type: {out.get('doc_type')} | rewrites: {out.get('attempts', 0)} "
        f"| cache_hit={out.get('cache_hit')} | early_fired={out.get('early_fired')} "
        f"| bg_pages={out.get('background_pages')} ==="
    )
    if out.get("web_skipped_reason"):
        print(f"\n=== Web skipped: {out['web_skipped_reason']} ===")
    else:
        print(
            f"\n=== Web: {len(out.get('web_docs') or [])} docs | "
            f"session={out.get('crawl_session_id')} ==="
        )
    print(f"\n=== DB docs: {len(out.get('docs') or [])} ===")
    print("\n=== Answer ===")
    print(out.get("answer"))
    print("\n=== Citations ===")
    for c in out.get("citations") or []:
        print(f"  - {c}")
