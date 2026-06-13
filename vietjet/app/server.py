from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from vietjet.agents.combined_agent import (
    REQUIRED_SLOTS,
    CombinedState,
    _is_slot_filled,
    get_graph,
)
from vietjet.crawlers import CrawlCoordinator
from vietjet.agents.qna_agentic import _initial_state as _qna_initial_state
from vietjet.agents.qna_agentic import get_graph as _get_qna_graph
from vietjet.app.timing import log_api_time

_sessions: Dict[str, CombinedState] = {}

def _new_state() -> CombinedState:
    return {
        "slots": {},
        "attempts": 0,
        "done": False,
        "user_input": "",
        "question": "",
        "intent": None,
        "answer": "",
        "slot_question": "",
        "web_candidates": [],
        "web_chosen_urls": [],
        "web_docs": [],
        "web_skipped_reason": None,
        "merged_docs": [],
        "cache_hit": False,
        "early_fired": False,
        "crawl_session_id": None,
        "background_pages": 0,
    }


def _missing_list(slots: dict) -> List[str]:
    return [s for s in REQUIRED_SLOTS if not _is_slot_filled(slots, s)]


class ThreadResponse(BaseModel):
    thread_id: str
    message: str


class ChatRequest(BaseModel):
    thread_id: str
    message: str


class ChatResponse(BaseModel):
    thread_id: str
    done: bool
    intent: Optional[str] = None
    answer: str = ""
    slot_question: str = ""
    missing_slots: List[str] = []
    slots: Dict[str, Any] = {}
    citations: List[str] = []
    is_off_topic: bool = False
    escalate: bool = False
    web_chosen_urls: List[str] = []
    web_skipped_reason: Optional[str] = None
    # Parallel crawl metadata
    cache_hit: bool = False
    early_fired: bool = False
    crawl_session_id: Optional[str] = None
    background_pages: int = 0


app = FastAPI(title="Vietjet Combined Agent", version="1.0.0")

# build graph & sinh ảnh ngay khi import
_graph = get_graph()
_qna_graph = _get_qna_graph()  # standalone QnA graph (no intent / no slots)

# Singleton coordinator dùng chung cho mọi request /qa-stream
_coordinator: CrawlCoordinator | None = None


def _get_coordinator() -> CrawlCoordinator:
    global _coordinator
    if _coordinator is None:
        _coordinator = CrawlCoordinator()
    return _coordinator


@app.post("/thread", response_model=ThreadResponse)
def create_thread() -> ThreadResponse:
    tid = str(uuid.uuid4())
    _sessions[tid] = _new_state()
    return ThreadResponse(
        thread_id=tid,
        message=(
            "Xin chào! Mình là trợ lý Vietjet. "
            "Bạn có thể hỏi quy định/giá vé/hành lý (câu hỏi) "
            "hoặc yêu cầu thao tác như đổi vé, hoàn vé, sửa tên... (yêu cầu)."
        ),
    )


def _next_thread_state(prev: CombinedState) -> CombinedState:
    """Nếu lượt trước đã done, mở session mới (reset state) để hỏi lượt mới."""
    if prev.get("done"):
        fresh = _new_state()
        return fresh
    return prev


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    state = _sessions.get(req.thread_id)
    if state is None:
        raise HTTPException(status_code=404, detail="thread_id không tồn tại")

    state = _next_thread_state(state)
    state["user_input"] = (req.message or "").strip()

    new_state: CombinedState = await _graph.ainvoke(state)
    _sessions[req.thread_id] = new_state

    return ChatResponse(
        thread_id=req.thread_id,
        done=bool(new_state.get("done")),
        intent=new_state.get("intent"),
        answer=new_state.get("answer", "") or "",
        slot_question=new_state.get("slot_question", "") or "",
        missing_slots=_missing_list(new_state.get("slots") or {}),
        slots=new_state.get("slots", {}) or {},
        citations=new_state.get("citations", []) or [],
        is_off_topic=bool(new_state.get("is_off_topic")),
        escalate=bool(new_state.get("escalate")),
        web_chosen_urls=new_state.get("web_chosen_urls", []) or [],
        web_skipped_reason=new_state.get("web_skipped_reason"),
        cache_hit=bool(new_state.get("cache_hit")),
        early_fired=bool(new_state.get("early_fired")),
        crawl_session_id=new_state.get("crawl_session_id"),
        background_pages=int(new_state.get("background_pages") or 0),
    )


@app.get("/thread/{thread_id}")
def get_thread(thread_id: str) -> Dict[str, Any]:
    state = _sessions.get(thread_id)
    if state is None:
        raise HTTPException(status_code=404, detail="thread_id không tồn tại")
    _drop = {"docs", "web_docs", "merged_docs"}
    clean = {k: v for k, v in state.items() if k not in _drop}
    return clean


@app.delete("/thread/{thread_id}")
def delete_thread(thread_id: str) -> Dict[str, str]:
    _sessions.pop(thread_id, None)
    return {"status": "deleted", "thread_id": thread_id}


# ---------------------------------------------------------------------------
# /qna — Standalone QnA endpoint (qna_agentic graph: RAG + parallel crawl)
# ---------------------------------------------------------------------------
class QnaRequest(BaseModel):
    question: str
    web_search: bool = False


class QnaResponse(BaseModel):
    question: str
    answer: str = ""
    citations: List[str] = []
    doc_type: Optional[str] = None
    attempts: int = 0
    sufficient: bool = False
    db_docs_count: int = 0
    web_docs_count: int = 0
    web_chosen_urls: List[str] = []
    web_skipped_reason: Optional[str] = None
    cache_hit: bool = False
    early_fired: bool = False
    crawl_session_id: Optional[str] = None
    background_pages: int = 0
    context_docs: List[Dict[str, Any]] = []


def _doc_to_context(d: Any, origin: str) -> Dict[str, Any]:
    md = getattr(d, "metadata", {}) or {}
    return {
        "id": md.get("id"),
        "origin": origin,
        "source": md.get("source"),
        "section_path": md.get("section_path"),
        "doc_type": md.get("doc_type"),
        "content": getattr(d, "page_content", "") or "",
    }


def _collect_context_docs(out: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs = [_doc_to_context(d, "db") for d in (out.get("docs") or [])]
    docs += [_doc_to_context(d, "web") for d in (out.get("web_docs") or [])]
    return docs


def _qna_to_response(question: str, out: Dict[str, Any]) -> QnaResponse:
    return QnaResponse(
        question=question,
        answer=out.get("answer", "") or "",
        citations=out.get("citations") or [],
        doc_type=out.get("doc_type"),
        attempts=int(out.get("attempts") or 0),
        sufficient=bool(out.get("sufficient")),
        db_docs_count=len(out.get("docs") or []),
        web_docs_count=len(out.get("web_docs") or []),
        web_chosen_urls=out.get("web_chosen_urls") or [],
        web_skipped_reason=out.get("web_skipped_reason"),
        cache_hit=bool(out.get("cache_hit")),
        early_fired=bool(out.get("early_fired")),
        crawl_session_id=out.get("crawl_session_id"),
        background_pages=int(out.get("background_pages") or 0),
        context_docs=_collect_context_docs(out),
    )


@app.post("/qna", response_model=QnaResponse)
@log_api_time("/qna")
async def qna(req: QnaRequest) -> QnaResponse:
    """One-shot QnA qua qna_agentic graph.

    Khác `/chat`: không có classify_intent / slot-filling. Luôn chạy luồng
    RAG + parallel crawl. Phù hợp khi client biết chắc đây là câu hỏi.
    """
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question empty")
    out = await _qna_graph.ainvoke(_qna_initial_state(question, req.web_search))
    return _qna_to_response(question, out)


@app.post("/qna-stream")
async def qna_stream(req: QnaRequest):
    """SSE stream cho qna_agentic — emit per-node update + final.

    Event types:
      - route / db_retrieve / parallel_crawl / merge / grade / rewrite / generate
        — state diff (slim, không chứa Documents nặng)
      - final — QnaResponse đầy đủ
      - error
    """
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question empty")

    def _slim(diff: dict) -> dict:
        out: dict = {}
        for k, v in diff.items():
            if k in ("docs", "web_docs", "merged_docs"):
                if isinstance(v, list):
                    out[f"{k}_count"] = len(v)
                continue
            try:
                json.dumps(v, ensure_ascii=False)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = repr(v)[:200]
        return out

    async def event_gen():
        init = _qna_initial_state(question, req.web_search)
        final_state: Dict[str, Any] = dict(init)
        try:
            async for step in _qna_graph.astream(init, stream_mode="updates"):
                for node_name, diff in step.items():
                    if not isinstance(diff, dict):
                        continue
                    final_state.update(diff)
                    payload = json.dumps(_slim(diff), ensure_ascii=False)
                    yield f"event: {node_name}\ndata: {payload}\n\n"
            final = _qna_to_response(question, final_state)
            yield f"event: final\ndata: {final.model_dump_json()}\n\n"
        except Exception as exc:
            err = json.dumps({"reason": str(exc)}, ensure_ascii=False)
            yield f"event: error\ndata: {err}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# /qa-stream — SSE endpoint cho parallel crawl agent (PLAN_PARALLEL_CRAWL_AGENT)
# ---------------------------------------------------------------------------
class StreamRequest(BaseModel):
    query: str
    home_urls: Optional[List[str]] = None


@app.post("/qa-stream")
async def qa_stream(req: StreamRequest):
    """SSE stream events từ CrawlCoordinator (KHÔNG qua LangGraph).

    Dùng khi client muốn raw crawl events. Để có final answer (qua graph QnA),
    xem `/chat-stream`.

    Event types:
      - cache_hit:      { docs, count }
      - partial_answer: { results, reason, early_fired, session_id }
      - ingested:       { pages, chunks }
      - done:           { session_id, frontier, judge_collected }
      - error:          { reason }
    """
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query empty")

    coord = _get_coordinator()
    if req.home_urls:
        coord = CrawlCoordinator(home_urls=req.home_urls)

    async def event_gen():
        try:
            async for ev in coord.stream(query):
                payload = json.dumps(ev.payload, ensure_ascii=False)
                yield f"event: {ev.type}\ndata: {payload}\n\n"
        except Exception as exc:
            err = json.dumps({"reason": str(exc)}, ensure_ascii=False)
            yield f"event: error\ndata: {err}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# /chat-stream — SSE wrapper cho /chat (LangGraph + Parallel Crawl)
# ---------------------------------------------------------------------------
@app.post("/chat-stream")
async def chat_stream(req: ChatRequest):
    """SSE: stream từng bước của graph cho user — node-by-node update.

    Mỗi node hoàn thành sẽ emit 1 event với name = tên node, payload =
    state diff. Đặc biệt:
      - intent       — sau classify_intent
      - parallel_crawl — sau qna_parallel_crawl (cache_hit/early_fired/...)
      - generate     — sau qna_generate (final answer)
    Cuối cùng emit `final` (ChatResponse đầy đủ).
    """
    state = _sessions.get(req.thread_id)
    if state is None:
        raise HTTPException(status_code=404, detail="thread_id không tồn tại")

    state = _next_thread_state(state)
    state["user_input"] = (req.message or "").strip()

    # Map node-name → event-name xuất ra FE (chỉ những node có ý nghĩa)
    EVENT_MAP = {
        "classify_intent":   "intent",
        "qna_route":         "route",
        "qna_parallel_crawl": "parallel_crawl",
        "qna_db_retrieve":   "db_retrieve",
        "qna_merge":         "merge",
        "qna_grade":         "grade",
        "qna_rewrite":       "rewrite",
        "qna_generate":      "generate",
        "extract_entity":    "extract_entity",
        "request_slot":      "request_slot",
        "query_request":     "query_request",
        "generate_request":  "generate_request",
        "off_topic":         "off_topic",
        "escalate":          "escalate",
    }

    def _slim(diff: dict) -> dict:
        """Loại bỏ field nặng (Documents) để payload SSE nhẹ + JSON-safe."""
        out: dict = {}
        for k, v in diff.items():
            if k in ("docs", "web_docs", "merged_docs", "user_input"):
                if isinstance(v, list):
                    out[f"{k}_count"] = len(v)
                continue
            try:
                json.dumps(v, ensure_ascii=False)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = repr(v)[:200]
        return out

    async def event_gen():
        final_state: CombinedState = dict(state)  # accumulate diffs
        try:
            async for step in _graph.astream(state, stream_mode="updates"):
                # step = {node_name: state_diff}
                for node_name, diff in step.items():
                    if not isinstance(diff, dict):
                        continue
                    final_state.update(diff)
                    ev_name = EVENT_MAP.get(node_name, node_name)
                    payload = json.dumps(_slim(diff), ensure_ascii=False)
                    yield f"event: {ev_name}\ndata: {payload}\n\n"

            _sessions[req.thread_id] = final_state
            final = ChatResponse(
                thread_id=req.thread_id,
                done=bool(final_state.get("done")),
                intent=final_state.get("intent"),
                answer=final_state.get("answer", "") or "",
                slot_question=final_state.get("slot_question", "") or "",
                missing_slots=_missing_list(final_state.get("slots") or {}),
                slots=final_state.get("slots", {}) or {},
                citations=final_state.get("citations", []) or [],
                is_off_topic=bool(final_state.get("is_off_topic")),
                escalate=bool(final_state.get("escalate")),
                web_chosen_urls=final_state.get("web_chosen_urls", []) or [],
                web_skipped_reason=final_state.get("web_skipped_reason"),
                cache_hit=bool(final_state.get("cache_hit")),
                early_fired=bool(final_state.get("early_fired")),
                crawl_session_id=final_state.get("crawl_session_id"),
                background_pages=int(final_state.get("background_pages") or 0),
            )
            yield f"event: final\ndata: {final.model_dump_json()}\n\n"
        except Exception as exc:
            err = json.dumps({"reason": str(exc)}, ensure_ascii=False)
            yield f"event: error\ndata: {err}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run("vietjet.app.server:app", host="127.0.0.1", port=8002, reload=False)
