"""JudgeConsumer: chấm content stream từ CrawlAgent, fire early-answer khi đạt threshold.

Pipeline 4 bước cho mỗi page:
  1. Extract snippet quanh keyword (rẻ, không LLM)
  2. Embedding cosine sim giữa query và snippet
  3. Nếu sim ≥ MED → gọi LLM rate confidence (high/medium/low)
  4. Nếu confidence=high AND sim ≥ HIGH → fire early-answer event,
     set mode_ref["mode"]="background" → CrawlAgent đổi target queue

JudgeConsumer KHÔNG dừng sau khi fire — nó tiếp tục consume page còn lại để
gom thêm doc cho generate (rerank-friendly).
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Callable

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from vietjet.config import (
    LLM_MODEL,
    PARALLEL_JUDGE_SIM_HIGH,
    PARALLEL_JUDGE_SIM_MED,
    PARALLEL_SNIPPET_WINDOW,
)
from vietjet.crawl_parallel.agent import PageItem


_PROMPT = """Bạn đánh giá đoạn web có chứa câu trả lời cho câu hỏi không.

CÂU HỎI: {query}

ĐOẠN WEB:
{snippet}

Trả về CHỈ JSON đúng format (không markdown, không text khác):
{{"confidence": "high"|"medium"|"low", "reason": "<1 câu>"}}"""


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


def _tokenize_keywords(query: str) -> list[str]:
    return [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]


def _extract_snippet(markdown: str, query: str, window: int = PARALLEL_SNIPPET_WINDOW) -> str:
    """Tìm window quanh keyword đầu tiên match. Fallback: phần đầu doc."""
    if not markdown:
        return ""
    md_low = markdown.lower()
    kws = _tokenize_keywords(query)
    best_pos = -1
    for kw in kws:
        pos = md_low.find(kw)
        if pos != -1 and (best_pos == -1 or pos < best_pos):
            best_pos = pos
    if best_pos == -1:
        return markdown[:window]
    start = max(0, best_pos - window // 4)
    end = min(len(markdown), start + window)
    return markdown[start:end]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class JudgeResult:
    __slots__ = ("url", "snippet", "sim", "confidence", "markdown", "title")

    def __init__(
        self,
        url: str,
        snippet: str,
        sim: float,
        confidence: str,
        markdown: str,
        title: str = "",
    ) -> None:
        self.url = url
        self.snippet = snippet
        self.sim = sim
        self.confidence = confidence
        self.markdown = markdown
        self.title = title

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "snippet": self.snippet[:200],
            "sim": round(self.sim, 4),
            "confidence": self.confidence,
            "title": self.title,
        }


class JudgeConsumer:
    def __init__(
        self,
        query: str,
        query_embed: np.ndarray,
        judge_queue: asyncio.Queue,
        ingest_queue: asyncio.Queue,
        mode_ref: dict,
        embedder: Callable[[str], np.ndarray],
        on_early_answer: Callable[[list[JudgeResult]], Any] | None = None,
        llm_model: str = LLM_MODEL,
        sim_high: float = PARALLEL_JUDGE_SIM_HIGH,
        sim_med: float = PARALLEL_JUDGE_SIM_MED,
    ) -> None:
        self.query = query
        self.query_embed = query_embed
        self.judge_queue = judge_queue
        self.ingest_queue = ingest_queue
        self.mode_ref = mode_ref
        self.embedder = embedder
        self.on_early_answer = on_early_answer
        self.sim_high = sim_high
        self.sim_med = sim_med
        self._llm = ChatOllama(model=llm_model, temperature=0.0)
        self.collected: list[JudgeResult] = []
        self.early_fired: bool = False

    async def run(self) -> None:
        from vietjet.crawl_parallel.agent import SENTINEL

        while True:
            item = await self.judge_queue.get()
            if item is SENTINEL:
                print(f"[judge] sentinel received, collected={len(self.collected)}")
                return
            if not isinstance(item, PageItem):
                continue
            try:
                await self._process(item)
            except Exception as exc:
                print(f"[judge] error on {item.url}: {exc}")

    async def _process(self, page: PageItem) -> None:
        snippet = _extract_snippet(page.markdown, self.query)
        if not snippet.strip():
            return

        # Cheap path: embedding sim
        snippet_embed = await asyncio.to_thread(self.embedder, snippet)
        sim = _cosine(self.query_embed, snippet_embed)

        confidence = "low"
        if sim >= self.sim_med:
            confidence = await self._llm_rate(snippet)

        result = JudgeResult(
            url=page.url,
            snippet=snippet,
            sim=sim,
            confidence=confidence,
            markdown=page.markdown,
            title=page.title,
        )
        self.collected.append(result)
        print(
            f"[judge] page={page.url} sim={sim:.3f} conf={confidence} "
            f"(early_fired={self.early_fired})"
        )

        # Early-answer trigger
        if (
            not self.early_fired
            and confidence == "high"
            and sim >= self.sim_high
        ):
            self.early_fired = True
            self.mode_ref["mode"] = "background"
            print(f"[judge] EARLY-ANSWER fired @ {page.url}")
            # Sau khi switch mode, page cũ đã có trong collected sẽ cũng cần ingest
            # → đẩy luôn vào ingest_queue
            await self._spill_to_ingest()
            if self.on_early_answer is not None:
                try:
                    res = self.on_early_answer(self.collected)
                    if asyncio.iscoroutine(res):
                        await res
                except Exception as exc:
                    print(f"[judge] on_early_answer error: {exc}")

    async def _spill_to_ingest(self) -> None:
        """Khi chuyển sang background mode, các page đã judge cũng nên ingest."""
        for r in self.collected:
            page = PageItem(
                url=r.url, markdown=r.markdown, title=r.title, agent_id=-1
            )
            try:
                self.ingest_queue.put_nowait(page)
            except asyncio.QueueFull:
                await self.ingest_queue.put(page)

    async def _llm_rate(self, snippet: str) -> str:
        try:
            msg = await self._llm.ainvoke([
                SystemMessage(content="Bạn là người đánh giá độ liên quan của đoạn web với câu hỏi. Trả JSON đúng format."),
                HumanMessage(content=_PROMPT.format(query=self.query, snippet=snippet[:1500])),
            ])
            data = _extract_json(msg.content if hasattr(msg, "content") else str(msg))
            conf = (data.get("confidence") or "low").strip().lower()
            if conf not in ("high", "medium", "low"):
                conf = "low"
            return conf
        except Exception as exc:
            print(f"[judge] llm_rate error: {exc}")
            return "low"
