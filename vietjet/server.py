"""FastAPI server cho Vietjet combined agent.

Endpoints:
- POST /thread          → khởi tạo session, trả thread_id + prompt khởi đầu.
- POST /chat            → gửi message vào graph, server giữ state theo thread_id.
- GET  /thread/{id}     → debug, xem state hiện tại.
- DELETE /thread/{id}   → xoá session.

Chạy:
    uvicorn vietjet.server:app --host 127.0.0.1 --port 8002 --reload
hoặc:
    python -m vietjet.server
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from vietjet.combined_agent import (
    REQUIRED_SLOTS,
    CombinedState,
    _is_slot_filled,
    get_graph,
)


# ---------------------------------------------------------------------------
# In-memory session store (production nên dùng Redis hoặc langgraph checkpointer)
# ---------------------------------------------------------------------------
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
    }


def _missing_list(slots: dict) -> List[str]:
    return [s for s in REQUIRED_SLOTS if not _is_slot_filled(slots, s)]


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Vietjet Combined Agent", version="1.0.0")

# build graph & sinh ảnh ngay khi import
_graph = get_graph()


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
    )


@app.get("/thread/{thread_id}")
def get_thread(thread_id: str) -> Dict[str, Any]:
    state = _sessions.get(thread_id)
    if state is None:
        raise HTTPException(status_code=404, detail="thread_id không tồn tại")
    # docs là Document objects → bỏ ra cho JSON-friendly
    clean = {k: v for k, v in state.items() if k != "docs"}
    return clean


@app.delete("/thread/{thread_id}")
def delete_thread(thread_id: str) -> Dict[str, str]:
    _sessions.pop(thread_id, None)
    return {"status": "deleted", "thread_id": thread_id}


if __name__ == "__main__":
    uvicorn.run("vietjet.server:app", host="127.0.0.1", port=8002, reload=False)
