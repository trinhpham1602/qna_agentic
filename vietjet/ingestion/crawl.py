"""Stage 1: crawl Vietjet pages with Firecrawl (async + retry)."""

from __future__ import annotations
import asyncio
from pathlib import Path

from firecrawl import AsyncFirecrawlApp

from vietjet.config import (
    CRAWL_MAX_RETRIES,
    CRAWL_RETRY_BACKOFF,
    FIRECRAWL_API_KEY,
    RAW_DIR,
    URLS,
)


async def _scrape_one(app: AsyncFirecrawlApp, item: dict, out_dir: Path) -> Path | None:
    url = item["url"]
    filename = item["filename"]
    out_path = out_dir / f"{filename}.md"

    last_err: Exception | None = None
    for attempt in range(1, CRAWL_MAX_RETRIES + 1):
        try:
            result = await app.scrape(
                url,
                formats=["markdown"],
                only_main_content=True,
                exclude_tags=["nav", "footer", "header", "aside", "script", "style", "form"],
                remove_base64_images=True,
            )
            markdown = getattr(result, "markdown", None)
            if not markdown:
                raise ValueError("empty markdown")
            out_path.write_text(markdown, encoding="utf-8")
            print(f"[OK]    {filename} (try {attempt}) → {out_path}")
            return out_path
        except Exception as e:
            last_err = e
            print(f"[RETRY] {filename} attempt {attempt}/{CRAWL_MAX_RETRIES}: {e}")
            if attempt < CRAWL_MAX_RETRIES:
                await asyncio.sleep(CRAWL_RETRY_BACKOFF * attempt)

    print(f"[FAIL]  {filename}: {last_err}")
    return None


async def crawl_all(items: list[dict] = URLS, out_dir: Path = RAW_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    app = AsyncFirecrawlApp(api_key=FIRECRAWL_API_KEY)
    results = await asyncio.gather(*(_scrape_one(app, it, out_dir) for it in items))
    return [p for p in results if p]


def main() -> None:
    asyncio.run(crawl_all())


if __name__ == "__main__":
    main()
