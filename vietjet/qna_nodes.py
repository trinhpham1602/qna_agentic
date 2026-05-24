from __future__ import annotations

import re

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
