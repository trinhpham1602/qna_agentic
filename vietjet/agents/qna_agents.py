from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

from firecrawl import AsyncFirecrawlApp
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from vietjet.config import (
    FIRECRAWL_API_KEY,
    LLM_MODEL,
    WEB_FETCH_CACHE_TTL,
    WEB_FETCH_CHUNK_CAP,
    WEB_FETCH_TIMEOUT,
    WEB_JUDGE_MAX_PICKS,
    WEB_SEARCH_INCLUDE_DOMAINS,
    WEB_SEARCH_LIMIT,
    WEB_SEARCH_TIMEOUT,
    WEB_SEARCH_URL_PREFIX,
)

_SYS_PROMPT_DIR = Path(__file__).resolve().parent.parent / "sys_prompt"

_llm_cache: dict[float, ChatOllama] = {}


def _llm(temperature: float = 0.0) -> ChatOllama:
    if temperature not in _llm_cache:
        _llm_cache[temperature] = ChatOllama(model=LLM_MODEL, temperature=temperature)
    return _llm_cache[temperature]


_prompt_cache: dict[str, str] = {}


def _load_prompt(name: str) -> str:
    if name not in _prompt_cache:
        _prompt_cache[name] = (_SYS_PROMPT_DIR / name).read_text(encoding="utf-8")
    return _prompt_cache[name]


def _extract_json(raw: str) -> dict:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        return json.loads(text)
    except Exception:
        return {}


_firecrawl: AsyncFirecrawlApp | None = None


def _firecrawl_app() -> AsyncFirecrawlApp:
    global _firecrawl
    if _firecrawl is None:
        _firecrawl = AsyncFirecrawlApp(api_key=FIRECRAWL_API_KEY)
    return _firecrawl


_fetch_cache: dict[str, tuple[float, str]] = {}


def _cache_get(url: str) -> str | None:
    entry = _fetch_cache.get(url)
    if entry is None:
        return None
    ts, val = entry
    if time.time() - ts > WEB_FETCH_CACHE_TTL:
        _fetch_cache.pop(url, None)
        return None
    return val


def _cache_set(url: str, value: str) -> None:
    _fetch_cache[url] = (time.time(), value)


def _attr(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _normalize_search_results(result) -> list[dict]:
    data = _attr(result, "data", None)
    if data is None:
        data = _attr(result, "web", None)
    if data is None and isinstance(result, list):
        data = result

    raw_list: list = []
    if isinstance(data, list):
        raw_list = data
    elif isinstance(data, dict):
        raw_list = data.get("web") or data.get("results") or []

    out: list[dict] = []
    for r in raw_list:
        url = _attr(r, "url")
        if not url:
            continue
        if WEB_SEARCH_URL_PREFIX and not url.startswith(WEB_SEARCH_URL_PREFIX):
            continue
        title = _attr(r, "title", "") or ""
        snippet = (
            _attr(r, "description", "")
            or _attr(r, "snippet", "")
            or _attr(r, "markdown", "")
            or ""
        )
        out.append({
            "url": url,
            "title": title,
            "snippet": str(snippet)[:300],
        })
    return out


async def web_search_agent(query: str) -> list[dict]:
    app = _firecrawl_app()
    kwargs: dict = {"limit": WEB_SEARCH_LIMIT}
    if WEB_SEARCH_INCLUDE_DOMAINS:
        kwargs["include_domains"] = list(WEB_SEARCH_INCLUDE_DOMAINS)
    result = await asyncio.wait_for(
        app.search(query, **kwargs),
        timeout=WEB_SEARCH_TIMEOUT,
    )
    return _normalize_search_results(result)


async def link_judge_agent(question: str, candidates: list[dict]) -> list[str]:
    if not candidates:
        return []
    sys_prompt = _load_prompt("judge_links")
    listing = []
    for i, c in enumerate(candidates, 1):
        listing.append(
            f"[{i}] url: {c['url']}\n    title: {c['title']}\n    snippet: {c['snippet']}"
        )
    user_msg = "CÂU HỎI: " + question + "\n\nDANH SÁCH ỨNG VIÊN:\n" + "\n\n".join(listing)

    try:
        resp = await _llm(0.0).ainvoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_msg),
        ])
        raw = resp.content if hasattr(resp, "content") else str(resp)
        data = _extract_json(raw)
        chosen = data.get("chosen") or []
        if not isinstance(chosen, list):
            chosen = []
        urls: list[str] = []
        for idx in chosen[:WEB_JUDGE_MAX_PICKS]:
            try:
                i = int(idx)
            except (ValueError, TypeError):
                continue
            if 1 <= i <= len(candidates):
                urls.append(candidates[i - 1]["url"])
        if urls:
            return urls
    except Exception:
        pass
    return [c["url"] for c in candidates[:WEB_JUDGE_MAX_PICKS]]


async def _scrape_one(app: AsyncFirecrawlApp, url: str) -> Document | None:
    cached = _cache_get(url)
    if cached is not None:
        return Document(
            page_content=cached,
            metadata={"source": url, "section_path": "web", "doc_type": "web", "id": url},
        )
    try:
        result = await asyncio.wait_for(
            app.scrape(
                url,
                formats=["markdown"],
                only_main_content=True,
                exclude_tags=["nav", "footer", "header", "aside", "script", "style", "form"],
                remove_base64_images=True,
            ),
            timeout=WEB_FETCH_TIMEOUT,
        )
    except Exception:
        return None
    markdown = (_attr(result, "markdown", "") or "")[:WEB_FETCH_CHUNK_CAP]
    if not markdown.strip():
        return None
    _cache_set(url, markdown)
    return Document(
        page_content=markdown,
        metadata={"source": url, "section_path": "web", "doc_type": "web", "id": url},
    )


async def web_fetch_agent(urls: list[str]) -> list[Document]:
    if not urls:
        return []
    app = _firecrawl_app()
    docs = await asyncio.gather(*(_scrape_one(app, u) for u in urls))
    return [d for d in docs if d is not None]


def _format_docs(docs: list[Document], label: str, cap: int = 1200) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "?")
        sec = d.metadata.get("section_path", "?")
        parts.append(f"[{label}-{i}] {src} > {sec}\n{d.page_content[:cap]}")
    return "\n\n".join(parts)


async def grade_agent(question: str, docs: list[Document]) -> bool:
    if not docs:
        return False
    sys_prompt = _load_prompt("qna_grade")
    user_msg = (
        f"CÂU HỎI:\n{question}\n\nTÀI LIỆU TRUY HỒI:\n{_format_docs(docs, 'DOC', cap=800)}"
    )
    try:
        resp = await _llm(0.0).ainvoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_msg),
        ])
        verdict = ((resp.content if hasattr(resp, "content") else str(resp)) or "").strip().lower()
        return verdict.startswith("y")
    except Exception:
        return True


async def rewrite_agent(question: str) -> str:
    sys_prompt = _load_prompt("qna_rewrite")
    try:
        resp = await _llm(0.2).ainvoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=question),
        ])
        raw = (resp.content if hasattr(resp, "content") else str(resp)) or ""
        new_q = raw.strip().splitlines()[0].strip(' "“”')
        return new_q or question
    except Exception:
        return question


async def generate_agent(
    question: str,
    db_docs: list[Document],
    web_docs: list[Document],
) -> tuple[str, list[str]]:
    sys_prompt = _load_prompt("qna_answer_web")
    sections = []
    if web_docs:
        sections.append(
            "NGUỒN WEB (ƯU TIÊN):\n" + _format_docs(web_docs, "WEB", cap=2000)
        )
    if db_docs:
        sections.append(
            "NGUỒN NỘI BỘ (DỰ PHÒNG):\n" + _format_docs(db_docs, "NỘI BỘ", cap=800)
        )
    context = "\n\n".join(sections) if sections else "(không có tài liệu)"
    user_msg = f"CÂU HỎI:\n{question}\n\nTÀI LIỆU:\n{context}\n\nTRẢ LỜI:"

    try:
        resp = await _llm(0.0).ainvoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_msg),
        ])
        answer = (resp.content if hasattr(resp, "content") else str(resp)) or ""
    except Exception:
        answer = (
            "Tôi không tìm thấy thông tin này trong tài liệu hiện có. "
            "Vui lòng liên hệ tổng đài 1900 1886."
        )

    citations: list[str] = []
    for d in web_docs:
        citations.append(f"web:{d.metadata.get('source', '?')}")
    for d in db_docs:
        src = d.metadata.get("source", "?")
        sec = d.metadata.get("section_path", "?")
        citations.append(f"db:{src}#{sec}")
    return answer.strip(), citations
