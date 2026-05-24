"""URLFrontier: shared async-safe URL queue với dedup.

Đảm bảo nhiều CrawlAgent không cùng đẩy/scrape cùng một URL. Dedup bằng set
được lock-guard; queue là asyncio.Queue thuần — backpressure ở producer khi
queue đầy.
"""

from __future__ import annotations

import asyncio


class URLFrontier:
    def __init__(self, max_size: int = 100) -> None:
        self._q: asyncio.Queue[str] = asyncio.Queue(maxsize=max_size)
        self._seen: set[str] = set()
        self._lock = asyncio.Lock()

    async def put(self, url: str) -> bool:
        """Push URL nếu chưa thấy. Trả về True nếu thật sự push, False nếu dup."""
        async with self._lock:
            if url in self._seen:
                return False
            self._seen.add(url)
        await self._q.put(url)
        return True

    async def get(self, timeout: float = 2.0) -> str | None:
        """Lấy URL kế tiếp. Trả None nếu queue rỗng sau timeout (idle stop)."""
        try:
            return await asyncio.wait_for(self._q.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def mark_seen(self, url: str) -> bool:
        """Mark URL là đã thấy nhưng KHÔNG push vào queue.

        Dùng cho URL mà Firecrawl đã tự crawl (qua watcher) — ta không cần
        scrape lại, nhưng phải nhớ là đã thấy để các agent khác bỏ qua.
        """
        if url in self._seen:
            return False
        self._seen.add(url)
        return True

    def has_seen(self, url: str) -> bool:
        return url in self._seen

    def stats(self) -> dict:
        return {"pending": self._q.qsize(), "seen": len(self._seen)}
