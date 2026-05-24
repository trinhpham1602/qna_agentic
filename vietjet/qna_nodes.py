from __future__ import annotations

import re

from langchain_core.documents import Document

from vietjet.config import MAX_REWRITES
from vietjet.qna_agents import (
    generate_agent,
    grade_agent,
    link_judge_agent,
    rewrite_agent,
    web_fetch_agent,
    web_search_agent,
)
from vietjet.qna_db import db_retrieve

_KW_TYPE = [
    (re.compile(r"\b(phí|lệ phí|giá|tiền|cost|vnd|usd|hóa đơn|vat)\b", re.I), "pricing"),
    (re.compile(r"\b(hành lý|baggage|kg|ký gửi|xách tay|quá khổ|đồ cấm|lithium|pin)\b", re.I), "baggage"),
    (re.compile(r"\b(bồi thường|delay|chậm|hủy chuyến|đền bù)\b", re.I), "compensation"),
    (re.compile(r"\b(cmnd|passport|hộ chiếu|giấy tờ|trẻ em|um|sky kids|thú cưng|sky pet)\b", re.I), "procedure"),
    (re.compile(r"\b(check[- ]?in|nối chuyến|thủ tục|làm thủ tục)\b", re.I), "procedure"),
    (re.compile(r"\b(thanh toán|momo|vnpay|hd saison|thẻ tín dụng|kênh)\b", re.I), "payment"),
    (re.compile(r"\b(skyboss|business|hạng thương gia|lounge|phòng chờ|deluxe|eco)\b", re.I), "service"),
    (re.compile(r"\b(điều kiện vé|điều lệ|hoàn vé|đổi vé|bảo lưu)\b", re.I), "regulation"),
]
_TABLE_HINT = re.compile(r"\b(bao nhiêu|giá|phí|kg|bảng|tỷ giá|mức)\b", re.I)


async def route_node(state: dict) -> dict:
    q = state["question"]
    doc_type: str | None = None
    for pat, t in _KW_TYPE:
        if pat.search(q):
            doc_type = t
            break
    return {
        "query": q,
        "doc_type": doc_type,
        "boost_tables": bool(_TABLE_HINT.search(q)),
        "attempts": 0,
        "web_skipped_reason": None,
    }


async def db_retrieve_node(state: dict) -> dict:
    doc_type = state.get("doc_type") if state.get("attempts", 0) == 0 else None
    docs = await db_retrieve(
        state["query"],
        doc_type=doc_type,
        boost_tables=state.get("boost_tables", False),
    )
    return {"docs": docs}


async def web_search_node(state: dict) -> dict:
    query = state["query"]
    try:
        candidates = await web_search_agent(query)
    except Exception as exc:
        print(f"[web_search] FAIL query={query!r} error={exc}")
        return {
            "web_candidates": [],
            "web_chosen_urls": [],
            "web_docs": [],
            "web_skipped_reason": f"search_error: {exc}",
        }

    print(f"[web_search] query={query!r} → {len(candidates)} candidates")
    for i, c in enumerate(candidates, 1):
        title = (c.get("title") or "").strip().replace("\n", " ")
        snippet = (c.get("snippet") or "").strip().replace("\n", " ")
        print(f"  [{i}] {c['url']}")
        if title:
            print(f"       title: {title[:120]}")
        if snippet:
            print(f"       snippet: {snippet[:160]}")
    return {"web_candidates": candidates, "web_skipped_reason": None}


async def link_judge_node(state: dict) -> dict:
    if state.get("web_skipped_reason"):
        return {"web_chosen_urls": []}
    candidates = state.get("web_candidates") or []
    if not candidates:
        print("[link_judge] no candidates → skip")
        return {"web_chosen_urls": [], "web_skipped_reason": "no_candidates"}
    urls = await link_judge_agent(state["question"], candidates)
    print(f"[link_judge] chose {len(urls)} URL(s) to crawl:")
    for i, u in enumerate(urls, 1):
        print(f"  → [{i}] {u}")
    return {"web_chosen_urls": urls}


async def web_fetch_node(state: dict) -> dict:
    if state.get("web_skipped_reason"):
        return {"web_docs": []}
    urls = state.get("web_chosen_urls") or []
    if not urls:
        return {"web_docs": []}
    try:
        docs = await web_fetch_agent(urls)
        return {"web_docs": docs}
    except Exception as exc:
        return {"web_docs": [], "web_skipped_reason": f"fetch_error: {exc}"}


async def parallel_crawl_node(state: dict) -> dict:
    """Node thay thế cho `web_search + link_judge + web_fetch`.

    Gọi CrawlCoordinator.stream():
      - cache_hit → dùng luôn docs từ DB web_live (skip Firecrawl)
      - partial_answer → gom JudgeResult thành web_docs
      - ingested / done → metadata

    Lazy import để tránh load CrawlCoordinator (heavy: Firecrawl + embedder)
    khi chỉ chạy nhánh request không có web search.
    """
    from vietjet.crawl_parallel.ask import _cache_doc_to_document, _result_to_document
    from vietjet.crawl_parallel.coordinator import CrawlCoordinator

    query = state.get("query") or state.get("question") or ""
    if not query.strip():
        return {
            "web_docs": [],
            "web_chosen_urls": [],
            "web_skipped_reason": "empty_query",
            "cache_hit": False,
            "early_fired": False,
            "crawl_session_id": None,
            "background_pages": 0,
        }

    coord = CrawlCoordinator()
    web_docs: list[Document] = []
    cache_hit = False
    early_fired = False
    session_id: str | None = None
    bg_pages = 0
    skipped_reason: str | None = None

    try:
        async for ev in coord.stream(query):
            if ev.type == "cache_hit":
                cache_hit = True
                for d in ev.payload.get("docs", []):
                    web_docs.append(_cache_doc_to_document(d))
                print(f"[parallel_crawl] CACHE HIT — {len(web_docs)} docs")
            elif ev.type == "partial_answer":
                early_fired = ev.payload.get("early_fired", False)
                session_id = ev.payload.get("session_id")
                reason = ev.payload.get("reason")
                results = ev.payload.get("results") or []
                if not results and not web_docs:
                    skipped_reason = f"no_results:{reason}"
                for r in results:
                    web_docs.append(_result_to_document(r))
                print(
                    f"[parallel_crawl] partial_answer reason={reason} "
                    f"early={early_fired} results={len(results)}"
                )
            elif ev.type == "ingested":
                bg_pages = ev.payload.get("pages", 0)
            elif ev.type == "done":
                session_id = session_id or ev.payload.get("session_id")
            elif ev.type == "error":
                skipped_reason = f"coord_error: {ev.payload.get('reason')}"
    except Exception as exc:
        print(f"[parallel_crawl] FAIL: {exc}")
        skipped_reason = f"crawl_error: {exc}"

    # Dedup theo source URL
    seen = set()
    deduped: list[Document] = []
    for d in web_docs:
        u = d.metadata.get("source") or d.metadata.get("id")
        if u in seen:
            continue
        seen.add(u)
        deduped.append(d)

    chosen_urls = [d.metadata.get("source") for d in deduped if d.metadata.get("source")]

    return {
        "web_docs": deduped,
        "web_chosen_urls": chosen_urls,
        "web_candidates": [],  # legacy field, để combined_agent không vỡ
        "web_skipped_reason": skipped_reason,
        "cache_hit": cache_hit,
        "early_fired": early_fired,
        "crawl_session_id": session_id,
        "background_pages": bg_pages,
    }


async def merge_node(state: dict) -> dict:
    db_docs = state.get("docs") or []
    web_docs = state.get("web_docs") or []
    return {"merged_docs": web_docs + db_docs}


async def grade_node(state: dict) -> dict:
    docs = state.get("merged_docs") or []
    sufficient = await grade_agent(state["question"], docs)
    return {"sufficient": sufficient}


async def rewrite_node(state: dict) -> dict:
    new_query = await rewrite_agent(state["question"])
    return {
        "query": new_query,
        "attempts": state.get("attempts", 0) + 1,
    }


async def generate_node(state: dict) -> dict:
    answer, citations = await generate_agent(
        state["question"],
        state.get("docs") or [],
        state.get("web_docs") or [],
    )
    return {"answer": answer, "citations": citations}


def after_grade(state: dict) -> str:
    if state.get("sufficient"):
        return "generate"
    if state.get("attempts", 0) < MAX_REWRITES:
        return "rewrite"
    return "generate"
