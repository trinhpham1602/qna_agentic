from __future__ import annotations

import asyncio

from firecrawl import AsyncFirecrawlApp

from vietjet.crawlers.text_clean import clean_ingest_text

_SCRAPE_OPTIONS = {
    "formats": ["markdown"],
    "only_main_content": True,
    "exclude_tags": ["nav", "footer", "header", "aside", "script", "style", "form"],
    "remove_base64_images": True,
}


def _extract_title(doc) -> str:
    md = getattr(doc, "metadata", None)
    if md is None:
        return ""
    title = getattr(md, "title", None) or getattr(md, "og_title", None)
    if not title and isinstance(md, dict):
        title = md.get("title") or md.get("og_title")
    return title or ""


async def firecrawl_fetch(
    app: AsyncFirecrawlApp, url: str, *, timeout: float
) -> tuple[str, str] | None:
    try:
        doc = await asyncio.wait_for(
            app.scrape(url, **_SCRAPE_OPTIONS), timeout=timeout
        )
    except asyncio.TimeoutError:
        print(f"[firecrawl-fetch] timeout url={url}")
        return None
    except Exception as exc:
        print(f"[firecrawl-fetch] failed url={url} err={exc}")
        return None

    markdown = clean_ingest_text(getattr(doc, "markdown", "") or "")
    if not markdown.strip():
        return None
    return markdown, _extract_title(doc)
