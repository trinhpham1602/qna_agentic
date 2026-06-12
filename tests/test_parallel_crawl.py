"""Integration tests cho vietjet/crawl_parallel.

Mock Firecrawl AsyncFirecrawlApp.watcher + start_crawl để chạy offline.
Verify:
  - URLFrontier dedup
  - JudgeConsumer fire early-answer khi sim đủ cao
  - CrawlCoordinator emit đúng sequence event (cache_miss → partial_answer → done)
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from vietjet.crawl_parallel.agent import PageItem, SENTINEL
from vietjet.crawl_parallel.frontier import URLFrontier
from vietjet.crawl_parallel.judge import (
    JudgeConsumer,
    _cosine,
    _extract_json,
    _extract_snippet,
)


# ─────────── unit tests pure funcs ────────────


def test_extract_snippet_finds_keyword():
    md = "lorem ipsum dolor.\n\nPhí ký gửi 20kg là 200.000 VND quốc nội.\n\ntail."
    s = _extract_snippet(md, "phí 20kg quốc nội", window=200)
    assert "20kg" in s.lower()


def test_extract_snippet_no_keyword_returns_head():
    md = "abcdef" * 100
    s = _extract_snippet(md, "totally unrelated", window=50)
    assert s == md[:50]


def test_cosine():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert _cosine(a, b) == 0.0
    assert _cosine(a, a) == 1.0
    assert _cosine(np.zeros(3), np.ones(3)) == 0.0


def test_extract_json_plain():
    assert _extract_json('{"confidence":"high"}') == {"confidence": "high"}


def test_extract_json_with_markdown_fence():
    raw = '```json\n{"confidence":"medium","reason":"x"}\n```'
    assert _extract_json(raw) == {"confidence": "medium", "reason": "x"}


def test_extract_json_with_surrounding_text():
    raw = 'My answer is: {"confidence":"low"} ok?'
    assert _extract_json(raw) == {"confidence": "low"}


def test_extract_json_invalid_returns_empty():
    assert _extract_json("not json at all") == {}


# ─────────── URLFrontier ────────────


def test_frontier_dedup():
    async def _run():
        f = URLFrontier(max_size=5)
        assert await f.put("u1") is True
        assert await f.put("u1") is False
        assert await f.put("u2") is True
        assert (await f.get(0.2)) == "u1"
        assert (await f.get(0.2)) == "u2"
        assert (await f.get(0.2)) is None
        assert f.has_seen("u1")
        assert not f.has_seen("u3")

    asyncio.run(_run())


def test_frontier_mark_seen_no_queue():
    async def _run():
        f = URLFrontier()
        assert f.mark_seen("x") is True
        assert f.mark_seen("x") is False
        # Không có gì trong queue
        assert (await f.get(0.1)) is None

    asyncio.run(_run())


# ─────────── JudgeConsumer ────────────


def test_judge_fires_early_when_sim_and_conf_high():
    async def _run():
        judge_q: asyncio.Queue = asyncio.Queue()
        ingest_q: asyncio.Queue = asyncio.Queue()
        mode = {"mode": "judge"}
        fired = []

        async def on_early(results):
            fired.append(len(results))

        # Embedder: page "good" trả vector trùng query (sim=1), page khác trả 0
        def fake_embed(text):
            return (
                np.ones(8, dtype=np.float32)
                if "good" in text.lower()
                else np.zeros(8, dtype=np.float32)
            )

        qe = np.ones(8, dtype=np.float32)
        j = JudgeConsumer("good query", qe, judge_q, ingest_q, mode, fake_embed, on_early_answer=on_early)

        # Patch LLM rate: trả "high" cho mọi page có sim >= MED
        async def fake_rate(self, snippet):
            return "high"
        with patch.object(JudgeConsumer, "_llm_rate", fake_rate):
            await judge_q.put(PageItem("u/bad", "bla bla", "", 0))
            await judge_q.put(PageItem("u/good", "this is a good match for query", "g", 0))
            await judge_q.put(SENTINEL)
            await j.run()

        assert j.early_fired is True
        assert mode["mode"] == "background"
        assert fired == [2]
        # Spill: cả 2 page collected phải vào ingest_q
        spilled = []
        while not ingest_q.empty():
            spilled.append(ingest_q.get_nowait())
        assert len(spilled) == 2

    asyncio.run(_run())


def test_judge_skips_llm_when_sim_below_med():
    async def _run():
        judge_q: asyncio.Queue = asyncio.Queue()
        ingest_q: asyncio.Queue = asyncio.Queue()
        mode = {"mode": "judge"}
        rate_calls = []

        async def fake_rate(self, snippet):
            rate_calls.append(snippet)
            return "high"

        # Always low sim → never call LLM
        def fake_embed(_text):
            return np.zeros(8, dtype=np.float32)

        with patch.object(JudgeConsumer, "_llm_rate", fake_rate):
            j = JudgeConsumer(
                "q", np.ones(8, dtype=np.float32),
                judge_q, ingest_q, mode, fake_embed, on_early_answer=None,
            )
            await judge_q.put(PageItem("u/1", "irrelevant content", "", 0))
            await judge_q.put(SENTINEL)
            await j.run()

        assert not j.early_fired
        assert rate_calls == [], "should skip LLM when sim < MED"
        assert mode["mode"] == "judge"

    asyncio.run(_run())


# ─────────── CrawlCoordinator integration với mock Firecrawl ────────────


class _FakeCrawlJob:
    def __init__(self, status, data):
        self.status = status
        self.data = data


def _make_fake_doc(url: str, markdown: str, title: str = ""):
    md = SimpleNamespace(url=url, title=title, source_url=url, og_title=None)
    return SimpleNamespace(metadata=md, markdown=markdown)


class _FakeAsyncFirecrawlApp:
    """Mock đủ để CrawlAgent chạy: start_crawl + watcher."""

    def __init__(self, pages_per_home: dict):
        self.pages_per_home = pages_per_home  # {home_url: [(url, markdown, title), ...]}
        self._next_id = 0
        self._jobs: dict[str, str] = {}  # job_id → home

    async def start_crawl(self, url, **kwargs):
        self._next_id += 1
        jid = f"job-{self._next_id}"
        self._jobs[jid] = url
        return SimpleNamespace(id=jid, url=url)

    def watcher(self, job_id, **kwargs):
        home = self._jobs[job_id]
        pages = self.pages_per_home.get(home, [])
        return _FakeWatcher(pages)


class _FakeWatcher:
    def __init__(self, pages):
        self.pages = pages

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        # Emit cumulative snapshots — mỗi yield thêm 1 page
        acc = []
        for url, md, title in self.pages:
            acc.append(_make_fake_doc(url, md, title))
            await asyncio.sleep(0)
            yield _FakeCrawlJob("scraping", list(acc))
        yield _FakeCrawlJob("completed", list(acc))


def test_coordinator_emits_partial_answer_with_mock_firecrawl():
    """End-to-end: 2 home URL, 3 pages mỗi home, 1 page match → early-answer fire."""

    async def _run():
        from vietjet.crawl_parallel.coordinator import CrawlCoordinator

        pages = {
            "https://home1/": [
                ("https://home1/p1", "irrelevant junk", ""),
                ("https://home1/p2", "this is a GOOD MATCH for our query", "good"),
            ],
            "https://home2/": [
                ("https://home2/p1", "lorem ipsum", ""),
            ],
        }
        fake_app = _FakeAsyncFirecrawlApp(pages)

        # Mock cache check → miss
        # Mock embedder: text chứa "good" → vector ones, else zeros.
        # Query "good query" cũng chứa "good" → embed cao,
        # page p2 chứa "GOOD MATCH" cũng cao → cosine=1.0.
        def fake_embed(text):
            return (
                np.ones(8, dtype=np.float32)
                if "good" in text.lower()
                else np.zeros(8, dtype=np.float32)
            )

        # Mock BackgroundIngest._get_store → no-op (không cần DB)
        from vietjet.crawl_parallel import background as bg_mod

        class _NoopStore:
            def add_documents(self, docs, ids=None):
                pass

        def fake_get_store(self):
            return _NoopStore()

        # Mock cache: always miss
        from vietjet.crawl_parallel import cache as cache_mod
        def fake_cache_check(self, query):
            return False, []

        # Mock LLM rate
        async def fake_rate(self, snippet):
            return "high" if "good" in snippet.lower() else "low"

        with patch.object(cache_mod.CacheChecker, "check", fake_cache_check), \
             patch.object(bg_mod.BackgroundIngest, "_get_store", fake_get_store), \
             patch("vietjet.crawl_parallel.coordinator._get_firecrawl", lambda: fake_app), \
             patch("vietjet.crawl_parallel.coordinator._embed_one", fake_embed), \
             patch.object(JudgeConsumer, "_llm_rate", fake_rate):
            coord = CrawlCoordinator(
                home_urls=["https://home1/", "https://home2/"],
                max_agents=2,
                max_pages=10,
                early_timeout=5.0,
                task_lifetime=10.0,
            )
            events = []
            async for ev in coord.stream("good query"):
                events.append(ev)

        types = [e.type for e in events]
        print("event types:", types)
        assert "partial_answer" in types
        assert "done" in types
        partial = next(e for e in events if e.type == "partial_answer")
        assert partial.payload["early_fired"] is True, partial.payload
        # URL "GOOD MATCH" page nằm trong results
        urls = [r["url"] for r in partial.payload["results"]]
        assert "https://home1/p2" in urls

    asyncio.run(_run())


if __name__ == "__main__":
    import sys
    # Run tests directly without pytest for quick smoke
    import inspect
    mod = sys.modules[__name__]
    failures = 0
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"OK   {name}")
        except Exception as exc:
            print(f"FAIL {name}: {exc}")
            failures += 1
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
