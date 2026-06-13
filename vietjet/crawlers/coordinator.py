"""CrawlCoordinator: orchestrate cache → multi-agent scrape (URL list) → judge → background ingest.

Flow:
  1. CacheChecker.check(query) — cache hit thì yield Event("cache_hit", docs) và END.
  2. Filter target URLs theo doc_type (DOC_TYPE_MAP) — chỉ crawl URL liên quan.
  3. Pre-fill frontier với target URLs. Spawn N CrawlAgent worker (asyncio.Task).
     Mỗi worker pull URL từ frontier → app.scrape() → push vào judge/ingest queue.
  4. JudgeConsumer chấm content song song. Khi đủ match → set early-answer event
     + flip mode_ref["mode"]="background".
  5. Yield Event("partial_answer", judge_results) ngay khi judge fire hoặc khi
     hết EARLY_ANSWER_TIMEOUT.
  6. Sau partial_answer, background_ingest tiếp tục consume tới khi hết queue
     hoặc MAX_TASK_LIFETIME.
  7. Yield Event("done").
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import numpy as np
from firecrawl import AsyncFirecrawlApp
from langchain_huggingface import HuggingFaceEmbeddings

from vietjet.config import (
    EMBED_MODEL,
    FIRECRAWL_API_KEY,
    PARALLEL_EARLY_ANSWER_TIMEOUT,
    PARALLEL_FRONTIER_MAX,
    PARALLEL_MAX_CONCURRENT_AGENTS,
    PARALLEL_MAX_PAGES_PER_QUERY,
    PARALLEL_MAX_TASK_LIFETIME,
    PARALLEL_SCRAPE_TIMEOUT,
)
from vietjet.crawlers.agent import SENTINEL, CrawlAgent
from vietjet.crawlers.background import BackgroundIngest
from vietjet.crawlers.cache_check import CacheChecker
from vietjet.crawlers.frontier import URLFrontier
from vietjet.crawlers.judge import JudgeConsumer, JudgeResult
from vietjet.crawlers.target_urls import filter_target_urls


@dataclass
class Event:
    type: str
    payload: dict = field(default_factory=dict)


_firecrawl: AsyncFirecrawlApp | None = None
_embedder: HuggingFaceEmbeddings | None = None


def _get_firecrawl() -> AsyncFirecrawlApp:
    global _firecrawl
    if _firecrawl is None:
        _firecrawl = AsyncFirecrawlApp(api_key=FIRECRAWL_API_KEY)
    return _firecrawl


def _get_embedder() -> HuggingFaceEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embedder


def _embed_one(text: str) -> np.ndarray:
    emb = _get_embedder().embed_query(text)
    return np.asarray(emb, dtype=np.float32)


class CrawlCoordinator:
    def __init__(
        self,
        *,
        max_agents: int = PARALLEL_MAX_CONCURRENT_AGENTS,
        max_pages: int = PARALLEL_MAX_PAGES_PER_QUERY,
        scrape_timeout: float = PARALLEL_SCRAPE_TIMEOUT,
        early_timeout: float = PARALLEL_EARLY_ANSWER_TIMEOUT,
        task_lifetime: float = PARALLEL_MAX_TASK_LIFETIME,
        cache_checker: CacheChecker | None = None,
    ) -> None:
        self.max_agents = max_agents
        self.max_pages = max_pages
        self.scrape_timeout = scrape_timeout
        self.early_timeout = early_timeout
        self.task_lifetime = task_lifetime
        self.cache = cache_checker or CacheChecker()

    async def stream(
        self, query: str, doc_type: str | None = None
    ) -> AsyncIterator[Event]:
        # ─── 1. Cache check ───────────────────────────────────────────
        hit, docs = await asyncio.to_thread(self.cache.check, query)
        if hit:
            yield Event("cache_hit", {
                "docs": [self._doc_to_dict(d) for d in docs],
                "count": len(docs),
            })
            yield Event("done", {"reason": "cache_hit"})
            return

        # ─── 2. Resolve target URLs ───────────────────────────────────
        target_urls = filter_target_urls(doc_type)
        target_urls = target_urls[: self.max_pages]
        if not target_urls:
            yield Event("partial_answer", {
                "results": [],
                "session_id": None,
                "reason": "no_target_urls",
                "early_fired": False,
            })
            yield Event("done", {"reason": "no_target_urls"})
            return

        # ─── 3. Setup queues + state ──────────────────────────────────
        session_id = uuid.uuid4().hex
        frontier = URLFrontier(max_size=max(PARALLEL_FRONTIER_MAX, len(target_urls) + 8))
        judge_q: asyncio.Queue = asyncio.Queue()
        ingest_q: asyncio.Queue = asyncio.Queue()
        mode_ref: dict[str, str] = {"mode": "judge"}
        early_event = asyncio.Event()

        async def _on_early(results: list[JudgeResult]) -> None:
            early_event.set()

        query_embed = await asyncio.to_thread(_embed_one, query)

        for url in target_urls:
            await frontier.put(url)
        print(
            f"[coord] frontier seeded with {len(target_urls)} URL(s) "
            f"(doc_type={doc_type})"
        )

        # ─── 4. Spawn agents + judge + background ingest ──────────────
        app = _get_firecrawl()
        agents = [
            CrawlAgent(
                agent_id=i,
                app=app,
                frontier=frontier,
                judge_queue=judge_q,
                ingest_queue=ingest_q,
                mode_ref=mode_ref,
                scrape_timeout=self.scrape_timeout,
            )
            for i in range(self.max_agents)
        ]
        judge = JudgeConsumer(
            query=query,
            query_embed=query_embed,
            judge_queue=judge_q,
            ingest_queue=ingest_q,
            mode_ref=mode_ref,
            embedder=_embed_one,
            on_early_answer=_on_early,
        )
        bg = BackgroundIngest(
            ingest_queue=ingest_q,
            session_id=session_id,
            source_query=query,
        )

        agent_tasks = [
            asyncio.create_task(a.run(), name=f"crawl-agent-{i}")
            for i, a in enumerate(agents)
        ]
        judge_task = asyncio.create_task(judge.run(), name="judge")
        bg_task = asyncio.create_task(bg.run(), name="bg-ingest")

        # ─── 5. Race: early-event vs agents-done ──────────────────────
        wait_early = asyncio.create_task(early_event.wait(), name="early-event")
        all_agents = asyncio.create_task(
            self._await_all(agent_tasks), name="agents-await-all"
        )

        async def _agents_then_drain_judge() -> None:
            await all_agents
            await judge_q.put(SENTINEL)
            try:
                await asyncio.wait_for(judge_task, timeout=10.0)
            except asyncio.TimeoutError:
                judge_task.cancel()

        drain_task = asyncio.create_task(_agents_then_drain_judge(), name="agents+drain")

        try:
            done, _pending = await asyncio.wait(
                {wait_early, drain_task},
                timeout=self.early_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
        except Exception as exc:
            yield Event("error", {"reason": f"wait_failed: {exc}"})
            done = set()

        if not wait_early.done():
            wait_early.cancel()

        # ─── 6. Emit partial_answer ──────────────────────────────────
        if early_event.is_set():
            reason = "early_match"
        elif drain_task in done:
            reason = "agents_finished"
        else:
            reason = "timeout"

        partial_payload = {
            "results": [r.to_dict() for r in judge.collected],
            "session_id": session_id,
            "reason": reason,
            "early_fired": judge.early_fired,
        }
        yield Event("partial_answer", partial_payload)

        # ─── 7. Background phase ─────────────────────────────────────
        mode_ref["mode"] = "background"
        if not judge_task.done():
            try:
                judge_q.put_nowait(SENTINEL)
            except asyncio.QueueFull:
                await judge_q.put(SENTINEL)

        remaining = max(0.0, self.task_lifetime - self.early_timeout)
        try:
            await asyncio.wait_for(all_agents, timeout=remaining)
        except asyncio.TimeoutError:
            print("[coord] agents lifetime exceeded → cancel")
            for t in agent_tasks:
                if not t.done():
                    t.cancel()
            try:
                await all_agents
            except Exception:
                pass

        if not drain_task.done():
            drain_task.cancel()
            try:
                await drain_task
            except Exception:
                pass

        await ingest_q.put(SENTINEL)

        try:
            await asyncio.wait_for(judge_task, timeout=5.0)
        except asyncio.TimeoutError:
            judge_task.cancel()
        except asyncio.CancelledError:
            pass

        try:
            await asyncio.wait_for(bg_task, timeout=remaining + 10.0)
        except asyncio.TimeoutError:
            bg_task.cancel()

        yield Event("ingested", {
            "pages": bg.ingested_pages,
            "chunks": bg.ingested_chunks,
        })
        yield Event("done", {
            "session_id": session_id,
            "frontier": frontier.stats(),
            "judge_collected": len(judge.collected),
        })

    @staticmethod
    async def _await_all(tasks: list[asyncio.Task]) -> None:
        await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _doc_to_dict(d) -> dict:
        return {
            "source": d.metadata.get("source"),
            "section_path": d.metadata.get("section_path"),
            "doc_type": d.metadata.get("doc_type"),
            "title": d.metadata.get("title"),
            "last_crawled_at": d.metadata.get("last_crawled_at"),
            "content": d.page_content[:600],
        }
