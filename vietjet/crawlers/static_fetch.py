from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
import trafilatura

from vietjet.cache.store import get_cache_store, stable_hash
from vietjet.config import (
    STATIC_FETCH_MIN_CHARS,
    STATIC_FETCH_TIMEOUT,
    STATIC_FETCH_USER_AGENT,
    TTL_WEB_PAGE_OFFICIAL,
)
from vietjet.crawlers.text_clean import clean_ingest_text

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            headers={"User-Agent": STATIC_FETCH_USER_AGENT},
            follow_redirects=True,
            timeout=STATIC_FETCH_TIMEOUT,
        )
    return _client


def _webpage_key(url: str) -> str:
    cache = get_cache_store()
    return cache.build_key("webpage", "v1", stable_hash(url))


def _extract_main_content(html: str) -> tuple[str, str]:
    extracted = trafilatura.extract(
        html,
        output_format="markdown",
        include_tables=True,
        include_links=False,
        favor_recall=True,
    )
    cleaned = clean_ingest_text(extracted or "")
    title = ""
    try:
        meta = trafilatura.extract_metadata(html)
        if meta is not None:
            title = meta.title or ""
    except Exception:
        title = ""
    return cleaned, title


async def static_fetch(url: str) -> tuple[str, str] | None:
    cache = get_cache_store()
    key = _webpage_key(url)
    cached = await cache.get_json(key)
    if cached and cached.get("content"):
        return cached["content"], cached.get("title") or ""

    try:
        resp = await _get_client().get(url)
    except Exception as exc:
        print(f"[static-fetch] request failed url={url} err={exc}")
        return None

    if resp.status_code >= 300:
        print(f"[static-fetch] status={resp.status_code} url={url}")
        return None
    if "html" not in resp.headers.get("content-type", "").lower():
        return None

    content, title = await asyncio.to_thread(_extract_main_content, resp.text)
    if len(content) < STATIC_FETCH_MIN_CHARS:
        return None

    await cache.set_json(
        key,
        {
            "url": url,
            "title": title,
            "content": content,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "content_hash": stable_hash(content),
        },
        ttl_seconds=TTL_WEB_PAGE_OFFICIAL,
    )
    return content, title
