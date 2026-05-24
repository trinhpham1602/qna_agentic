"""High-level helper: 1 câu hỏi → coordinator + DB retrieve + generate.

Dùng được cho:
- CLI / smoke test (non-stream): chờ coordinator yield hết, gom partial_answer
  + cache_hit, gọi generate_agent ra câu trả lời cuối.
- Tích hợp với LangGraph: wrap coordinator như 1 node thay cho web_search +
  link_judge + web_fetch trong graph cũ.

Trả về dict tương thích với combined_agent state:
  { "answer", "citations", "docs" (DB), "web_docs", "cache_hit", "session_id" }
"""

from __future__ import annotations

from langchain_core.documents import Document

from vietjet.crawl_parallel.coordinator import CrawlCoordinator, Event
from vietjet.qna_agents import generate_agent
from vietjet.qna_db import db_retrieve


def _result_to_document(r: dict) -> Document:
    return Document(
        page_content=r.get("snippet", ""),
        metadata={
            "source": r.get("url", "?"),
            "section_path": "web",
            "doc_type": "web_live",
            "title": r.get("title", ""),
            "sim": r.get("sim"),
            "confidence": r.get("confidence"),
            "id": r.get("url", "?"),
        },
    )


def _cache_doc_to_document(d: dict) -> Document:
    return Document(
        page_content=d.get("content", ""),
        metadata={
            "source": d.get("source", "?"),
            "section_path": d.get("section_path", "web"),
            "doc_type": d.get("doc_type", "web_live"),
            "title": d.get("title", ""),
            "last_crawled_at": d.get("last_crawled_at"),
            "id": d.get("source", "?"),
        },
    )


async def ask_parallel(
    question: str,
    *,
    coordinator: CrawlCoordinator | None = None,
    use_db: bool = True,
) -> dict:
    """One-shot: run coordinator → generate final answer.

    Khác `coordinator.stream()` ở chỗ đã wait đến event `done` và đã gọi
    `generate_agent`. Dùng cho REST endpoint không stream hoặc CLI.
    """
    coord = coordinator or CrawlCoordinator()
    web_docs: list[Document] = []
    cache_hit = False
    session_id: str | None = None
    early_fired = False
    bg_pages = 0

    async for ev in coord.stream(question):
        if ev.type == "cache_hit":
            cache_hit = True
            web_docs.extend(_cache_doc_to_document(d) for d in ev.payload.get("docs", []))
        elif ev.type == "partial_answer":
            early_fired = ev.payload.get("early_fired", False)
            session_id = ev.payload.get("session_id")
            for r in ev.payload.get("results", []):
                web_docs.append(_result_to_document(r))
        elif ev.type == "ingested":
            bg_pages = ev.payload.get("pages", 0)
        elif ev.type == "done":
            session_id = session_id or ev.payload.get("session_id")
        elif ev.type == "error":
            print(f"[ask_parallel] coordinator error: {ev.payload}")

    # Dedup web_docs theo URL (cache_hit và partial_answer có thể trùng)
    seen_url = set()
    deduped: list[Document] = []
    for d in web_docs:
        u = d.metadata.get("source")
        if u in seen_url:
            continue
        seen_url.add(u)
        deduped.append(d)
    web_docs = deduped

    db_docs: list[Document] = []
    if use_db:
        try:
            db_docs = await db_retrieve(question)
        except Exception as exc:
            print(f"[ask_parallel] db_retrieve failed: {exc}")

    answer, citations = await generate_agent(question, db_docs, web_docs)
    return {
        "answer": answer,
        "citations": citations,
        "docs": db_docs,
        "web_docs": web_docs,
        "cache_hit": cache_hit,
        "early_fired": early_fired,
        "session_id": session_id,
        "background_pages": bg_pages,
    }


if __name__ == "__main__":
    import asyncio
    import sys

    q = " ".join(sys.argv[1:]) or "phí ký gửi 20kg quốc nội"
    out = asyncio.run(ask_parallel(q))
    print("\n=== Question ===")
    print(q)
    print(f"\n=== Session: {out.get('session_id')} | cache_hit={out['cache_hit']} | "
          f"early_fired={out['early_fired']} | bg_pages={out['background_pages']} ===")
    print("\n=== Answer ===")
    print(out["answer"])
    print("\n=== Citations ===")
    for c in out.get("citations", []):
        print(f"  - {c}")
