from __future__ import annotations

import asyncio
from typing import Any

from firecrawl import AsyncFirecrawlApp

from vietjet.crawlers.firecrawl_fetch import firecrawl_fetch
from vietjet.crawlers.frontier import URLFrontier
from vietjet.crawlers.static_fetch import static_fetch


class PageItem:
    __slots__ = ("url", "markdown", "title", "agent_id", "method")

    def __init__(
        self,
        url: str,
        markdown: str,
        title: str = "",
        agent_id: int = 0,
        method: str = "static",
    ) -> None:
        self.url = url
        self.markdown = markdown
        self.title = title
        self.agent_id = agent_id
        self.method = method

    def __repr__(self) -> str:
        return (
            f"PageItem(url={self.url!r}, method={self.method}, "
            f"agent_id={self.agent_id}, md_len={len(self.markdown)})"
        )


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

    async def _enqueue_page(
        self, url: str, markdown: str, title: str, method: str
    ) -> None:
        if not markdown or not markdown.strip():
            return
        item = PageItem(
            url=url, markdown=markdown, title=title, agent_id=self.agent_id, method=method
        )
        await self._target_queue().put(item)
        self.emitted += 1

    async def _fetch_one(self, url: str) -> None:
        result = await static_fetch(url)
        method = "static"
        if result is None:
            result = await firecrawl_fetch(self.app, url, timeout=self.scrape_timeout)
            method = "firecrawl"
        if result is None:
            return
        markdown, title = result
        print(f"[crawl-agent {self.agent_id}] {method} url={url} md_len={len(markdown)}")
        await self._enqueue_page(url, markdown, title, method)

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
                await self._fetch_one(url)
        except asyncio.CancelledError:
            print(f"[crawl-agent {self.agent_id}] cancelled (emitted={self.emitted})")
            raise
