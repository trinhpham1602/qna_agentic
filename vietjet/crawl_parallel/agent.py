from __future__ import annotations

import asyncio
from typing import Any

from firecrawl import AsyncFirecrawlApp

from vietjet.crawl_parallel.frontier import URLFrontier


_SCRAPE_OPTIONS = {
    "formats": ["markdown"],
    "only_main_content": True,
    "exclude_tags": ["nav", "footer", "header", "aside", "script", "style", "form"],
    "remove_base64_images": True,
}


class PageItem:
    __slots__ = ("url", "markdown", "title", "agent_id")

    def __init__(self, url: str, markdown: str, title: str = "", agent_id: int = 0) -> None:
        self.url = url
        self.markdown = markdown
        self.title = title
        self.agent_id = agent_id

    def __repr__(self) -> str:
        return f"PageItem(url={self.url!r}, agent_id={self.agent_id}, md_len={len(self.markdown)})"


SENTINEL: Any = object()


class CrawlAgent:
    def __init__(
        self,
        agent_id: int,
        app: AsyncFirecrawlApp,
        frontier: URLFrontier,
        judge_queue: asyncio.Queue,
        ingest_queue: asyncio.Queue,
        mode_ref: dict,
        *,
        scrape_timeout: float = 30.0,
        idle_timeout: float = 2.0,
    ) -> None:
        self.agent_id = agent_id
        self.app = app
        self.frontier = frontier
        self.judge_queue = judge_queue
        self.ingest_queue = ingest_queue
        self.mode_ref = mode_ref
        self.scrape_timeout = scrape_timeout
        self.idle_timeout = idle_timeout
        self.emitted: int = 0
        self.error: str | None = None

    def _target_queue(self) -> asyncio.Queue:
        if self.mode_ref.get("mode") == "background":
            return self.ingest_queue
        return self.judge_queue

    async def _enqueue_page(self, url: str, markdown: str, title: str) -> None:
        if not markdown or not markdown.strip():
            return
        item = PageItem(url=url, markdown=markdown, title=title, agent_id=self.agent_id)
        await self._target_queue().put(item)
        self.emitted += 1

    async def _scrape_one(self, url: str) -> None:
        try:
            doc = await asyncio.wait_for(
                self.app.scrape(url, **_SCRAPE_OPTIONS),
                timeout=self.scrape_timeout,
            )
        except asyncio.TimeoutError:
            print(f"[crawl-agent {self.agent_id}] scrape timeout url={url}")
            return
        except Exception as exc:
            print(f"[crawl-agent {self.agent_id}] scrape failed url={url} err={exc}")
            return

        md = getattr(doc, "markdown", "") or ""
        title = self._extract_title(doc)
        await self._enqueue_page(url, md, title)

    async def run(self) -> None:
        try:
            while True:
                url = await self.frontier.get(timeout=self.idle_timeout)
                if url is None:
                    print(
                        f"[crawl-agent {self.agent_id}] frontier idle → stop "
                        f"(emitted={self.emitted})"
                    )
                    return
                await self._scrape_one(url)
        except asyncio.CancelledError:
            print(f"[crawl-agent {self.agent_id}] cancelled (emitted={self.emitted})")
            raise

    @staticmethod
    def _extract_title(doc) -> str:
        md = getattr(doc, "metadata", None)
        if md is None:
            return ""
        title = getattr(md, "title", None) or getattr(md, "og_title", None)
        if not title and isinstance(md, dict):
            title = md.get("title") or md.get("og_title")
        return title or ""
