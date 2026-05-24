"""CrawlAgent: stream page từ Firecrawl watcher → push vào judge/ingest queue.

Mỗi CrawlAgent đại diện cho 1 sub-agent search web (1 asyncio task), nhận 1
home URL, dùng start_crawl + watcher để stream từng page khi Firecrawl crawl xong.

Watcher của Firecrawl v2 yield CrawlJob snapshot **cumulative** — tức list .data
mỗi lần lớn dần. Agent so với set seen ở URLFrontier để chỉ emit URL mới.

Khi `mode_ref["mode"]` đổi từ `"judge"` sang `"background"`, page mới sẽ được
push vào `ingest_queue` thay vì `judge_queue`. Agent KHÔNG cancel task crawl
ngầm của Firecrawl — nó tự kết thúc khi đủ `limit`.
"""

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
    """Wrapper bare-minimum: url + markdown + title.

    Không dùng Firecrawl Document trực tiếp để giảm coupling với SDK version.
    """

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
        limit: int = 30,
        max_depth: int = 2,
        poll_interval: int = 2,
        allow_external: bool = False,
    ) -> None:
        self.agent_id = agent_id
        self.app = app
        self.frontier = frontier
        self.judge_queue = judge_queue
        self.ingest_queue = ingest_queue
        self.mode_ref = mode_ref
        self.limit = limit
        self.max_depth = max_depth
        self.poll_interval = poll_interval
        self.allow_external = allow_external
        self.emitted: int = 0
        self.error: str | None = None

    def _target_queue(self) -> asyncio.Queue:
        if self.mode_ref.get("mode") == "background":
            return self.ingest_queue
        return self.judge_queue

    async def _enqueue_page(self, url: str, markdown: str, title: str) -> None:
        if not markdown or not markdown.strip():
            return
        # dedup ở mức coordinator — nếu URL đã thấy bởi agent khác thì bỏ qua
        if not self.frontier.mark_seen(url):
            return
        item = PageItem(url=url, markdown=markdown, title=title, agent_id=self.agent_id)
        await self._target_queue().put(item)
        self.emitted += 1

    async def run(self, home_url: str) -> None:
        """Stream pages từ Firecrawl crawl job → push vào target queue."""
        try:
            resp = await self.app.start_crawl(
                home_url,
                limit=self.limit,
                max_discovery_depth=self.max_depth,
                allow_external_links=self.allow_external,
                scrape_options=_SCRAPE_OPTIONS,
            )
        except Exception as exc:
            self.error = f"start_crawl_failed: {exc}"
            print(f"[crawl-agent {self.agent_id}] {self.error}")
            return

        job_id = getattr(resp, "id", None)
        if not job_id:
            self.error = "no_job_id"
            return

        print(f"[crawl-agent {self.agent_id}] start home={home_url} job={job_id}")

        # Dedup theo URL trong cùng job (snapshot.data là cumulative)
        seen_in_job: set[str] = set()
        try:
            async for snapshot in self.app.watcher(
                job_id, kind="crawl", poll_interval=self.poll_interval
            ):
                for doc in snapshot.data or []:
                    url = self._extract_url(doc)
                    if not url or url in seen_in_job:
                        continue
                    seen_in_job.add(url)
                    md = getattr(doc, "markdown", "") or ""
                    title = self._extract_title(doc)
                    await self._enqueue_page(url, md, title)
                if snapshot.status in ("completed", "failed", "cancelled"):
                    print(
                        f"[crawl-agent {self.agent_id}] job {job_id} status={snapshot.status} "
                        f"emitted={self.emitted}"
                    )
                    return
        except asyncio.CancelledError:
            print(f"[crawl-agent {self.agent_id}] cancelled (emitted={self.emitted})")
            raise
        except Exception as exc:
            self.error = f"watcher_failed: {exc}"
            print(f"[crawl-agent {self.agent_id}] {self.error}")

    @staticmethod
    def _extract_url(doc) -> str | None:
        md = getattr(doc, "metadata", None)
        if md is None:
            return None
        # DocumentMetadata pydantic model: url field
        url = getattr(md, "url", None) or getattr(md, "source_url", None)
        if not url and isinstance(md, dict):
            url = md.get("url") or md.get("source_url")
        return url

    @staticmethod
    def _extract_title(doc) -> str:
        md = getattr(doc, "metadata", None)
        if md is None:
            return ""
        title = getattr(md, "title", None) or getattr(md, "og_title", None)
        if not title and isinstance(md, dict):
            title = md.get("title") or md.get("og_title")
        return title or ""
