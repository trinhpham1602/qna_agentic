"""Combined Vietjet agent — classify intent rồi route sang 2 nhánh.

Flow:
  classify_intent
       │
       ├── intent == "question" ─→ route → retrieve → grade ─→ (rewrite ↻) ─→ generate_qna ─→ END
       │                                                  └────────────────→ generate_qna ─→ END
       │
       └── intent == "request"  ─→ extract_entity ──→ off_topic ─→ END
                                                  └→ request_slot ─→ END (chờ user nhập tiếp)
                                                  └→ query_request ─→ generate_request ─→ END
                                                                  └→ escalate ─→ END

- "question" dùng RAG QnA hiện có của vietjet (vietjet/agent.py).
- "request" dùng slot-filling theo style review.py (4 slot: airline, ticket_class, route_type, group_policy)
  rồi truy vấn pgvector bảng qna_policy.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Literal, Optional, TypedDict

import psycopg2
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
import os
POLICY_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")


from config import (
    DB_CONNECTION_STRING as POLICY_DB,
)
from vietjet.agent import (
    grade_node as qna_grade_node,
    generate_node as qna_generate_node,
    retrieve_node as qna_retrieve_node,
    rewrite_node as qna_rewrite_node,
    route_node as qna_route_node,
    _after_grade,
)
from vietjet.config import LLM_MODEL


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYS_PROMPT_CLASSIFY = _PROJECT_ROOT / "sys_prompt" / "classify_intent"
SYS_PROMPT_EXTRACT = _PROJECT_ROOT / "sys_prompt" / "extract_required_slots"

REQUIRED_SLOTS = ["airline", "ticket_class", "route_type", "group_policy"]
POLICY_TABLE = "qna_policy"

SLOT_PROMPT = {
    "airline":      "Bạn đang hỏi về hãng bay nào? (vd: Vietjet)",
    "ticket_class": "Hạng vé là gì? (vd: eco, deluxe, skyboss, skyboss_business)",
    "route_type":   "Loại chặng là gì? (nội địa / quốc tế)",
    "group_policy": "Bạn cần thao tác nhóm nào? (vd: đổi vé, hoàn vé, hành lý, sửa tên, nâng hạng)",
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
Intent = Literal["question", "request"]


class CombinedState(TypedDict, total=False):
    # ---- chung ----
    user_input: str
    question: str
    intent: Optional[Intent]
    answer: str
    done: bool

    # ---- nhánh question (QnA) ----
    query: str
    doc_type: Optional[str]
    boost_tables: bool
    docs: List[Document]
    attempts: int
    sufficient: bool
    citations: List[str]

    # ---- nhánh request (slot-filling) ----
    slots: Dict[str, List[str]]
    next_slot: Optional[str]
    slot_question: str
    retrieved_rows: List[dict]
    is_off_topic: bool
    escalate: bool


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------
_llm_cache: Dict[float, ChatOllama] = {}


def _llm(temperature: float = 0.0) -> ChatOllama:
    if temperature not in _llm_cache:
        _llm_cache[temperature] = ChatOllama(model=LLM_MODEL, temperature=temperature)
    return _llm_cache[temperature]


_sys_prompt_classify: Optional[str] = None
_sys_prompt_extract: Optional[str] = None


def _load_text(path: Path, cache: Optional[str]) -> str:
    if cache is not None:
        return cache
    return path.read_text(encoding="utf-8")


def _load_classify_prompt() -> str:
    global _sys_prompt_classify
    _sys_prompt_classify = _load_text(SYS_PROMPT_CLASSIFY, _sys_prompt_classify)
    return _sys_prompt_classify


def _load_extract_prompt() -> str:
    global _sys_prompt_extract
    _sys_prompt_extract = _load_text(SYS_PROMPT_EXTRACT, _sys_prompt_extract)
    return _sys_prompt_extract


def _extract_json_obj(raw: str) -> dict:
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


# ---------------------------------------------------------------------------
# NODE 0: classify intent
# ---------------------------------------------------------------------------
async def classify_intent_node(state: CombinedState) -> CombinedState:
    user_input = (state.get("user_input") or "").strip()
    if user_input and not state.get("question"):
        state["question"] = user_input

    if not user_input:
        # đang ở giữa session, user gửi rỗng → giữ intent cũ nếu có
        state.setdefault("intent", None)
        return state

    sys_prompt = _load_classify_prompt()
    intent: Intent = "question"  # default an toàn
    try:
        resp = await _llm(0.0).ainvoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_input),
        ])
        data = _extract_json_obj(resp.content if hasattr(resp, "content") else str(resp))
        raw_intent = (data.get("intent") or "").strip().lower()
        if raw_intent in {"question", "request"}:
            intent = raw_intent  # type: ignore[assignment]
    except Exception:
        intent = "question"

    state["intent"] = intent
    return state


# ---------------------------------------------------------------------------
# QnA branch — reuse từ vietjet.agent
# Wrap route_node để set question/query đúng theo state combined
# ---------------------------------------------------------------------------
async def qna_route_wrapper(state: CombinedState) -> CombinedState:
    out = await qna_route_node({"question": state.get("question") or ""})
    state.update(out)
    return state


async def qna_retrieve_wrapper(state: CombinedState) -> CombinedState:
    out = await qna_retrieve_node(state)  # type: ignore[arg-type]
    state.update(out)
    return state


async def qna_grade_wrapper(state: CombinedState) -> CombinedState:
    out = await qna_grade_node(state)  # type: ignore[arg-type]
    state.update(out)
    return state


async def qna_rewrite_wrapper(state: CombinedState) -> CombinedState:
    out = await qna_rewrite_node(state)  # type: ignore[arg-type]
    state.update(out)
    return state


async def qna_generate_wrapper(state: CombinedState) -> CombinedState:
    out = await qna_generate_node(state)  # type: ignore[arg-type]
    state.update(out)
    state["done"] = True
    return state


# ---------------------------------------------------------------------------
# Request branch — slot-filling style review.py, viết lại async + dùng POLICY_DB
# ---------------------------------------------------------------------------
def _is_slot_filled(slots: dict, name: str) -> bool:
    v = slots.get(name)
    if isinstance(v, list):
        return len(v) > 0
    return bool(v)


def _missing_slot(slots: dict) -> Optional[str]:
    for s in REQUIRED_SLOTS:
        if not _is_slot_filled(slots, s):
            return s
    return None


def _parse_slot_json(raw: str) -> Dict[str, List[str]]:
    data = _extract_json_obj(raw)
    out: Dict[str, List[str]] = {}
    for s in REQUIRED_SLOTS:
        v = data.get(s, [])
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            v = []
        out[s] = [str(x).strip() for x in v if str(x).strip()]
    return out


async def extract_entity_node(state: CombinedState) -> CombinedState:
    user_input = (state.get("user_input") or "").strip()
    slots: Dict[str, List[str]] = dict(state.get("slots") or {})

    state["is_off_topic"] = False

    if not user_input:
        state["slots"] = slots
        state["next_slot"] = _missing_slot(slots)
        return state

    sys_prompt = _load_extract_prompt()
    extracted: Dict[str, List[str]] = {s: [] for s in REQUIRED_SLOTS}
    try:
        resp = await _llm(0.0).ainvoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_input),
        ])
        raw = resp.content if hasattr(resp, "content") else str(resp)
        extracted = _parse_slot_json(raw)
    except Exception:
        pass

    pre_empty = all(not _is_slot_filled(slots, s) for s in REQUIRED_SLOTS)
    for s in REQUIRED_SLOTS:
        if not _is_slot_filled(slots, s) and extracted.get(s):
            slots[s] = extracted[s]
    post_empty = all(not _is_slot_filled(slots, s) for s in REQUIRED_SLOTS)

    state["is_off_topic"] = pre_empty and post_empty
    state["slots"] = slots
    state["next_slot"] = _missing_slot(slots)
    state["user_input"] = ""
    return state


async def request_slot_node(state: CombinedState) -> CombinedState:
    slots = state.get("slots") or {}
    missing = [s for s in REQUIRED_SLOTS if not _is_slot_filled(slots, s)]
    if not missing:
        state["slot_question"] = ""
        return state
    if len(missing) == 1:
        state["slot_question"] = SLOT_PROMPT[missing[0]]
    else:
        lines = [f"- {SLOT_PROMPT[s]}" for s in missing]
        state["slot_question"] = (
            "Để xử lý yêu cầu của bạn, vui lòng cung cấp thêm các thông tin sau "
            "(có thể trả lời gộp 1 câu):\n" + "\n".join(lines)
        )
    state["done"] = False
    return state


def _pick_one(slots: dict, name: str) -> str:
    v = slots.get(name)
    if isinstance(v, list):
        return v[0] if v else ""
    return v or ""


def _vec_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


_policy_embedder: Optional[HuggingFaceEmbeddings] = None


def _get_policy_embedder() -> HuggingFaceEmbeddings:
    global _policy_embedder
    if _policy_embedder is None:
        _policy_embedder = HuggingFaceEmbeddings(model_name=POLICY_EMBED_MODEL)
    return _policy_embedder


def _query_policy_sync(slots: dict, question: str) -> List[dict]:
    embedder = _get_policy_embedder()
    airline      = _pick_one(slots, "airline")
    ticket_class = _pick_one(slots, "ticket_class")
    route_type   = _pick_one(slots, "route_type")
    group_policy = _pick_one(slots, "group_policy")

    query_text = question or (
        f"{group_policy} {airline} hạng vé {ticket_class} {route_type}"
    )
    qvec = embedder.embed_query(query_text)
    vec_lit = _vec_literal(qvec)
    sql = f"""
        SELECT policy_type, policy_desc, condition_decs, note, applied_pax_type,
               1 - (embedding <=> %s::vector) AS score
        FROM {POLICY_TABLE}
        WHERE airline = %s
          AND ticket_class = %s
          AND route_type = %s
          AND group_policy = %s
        ORDER BY embedding <=> %s::vector
        LIMIT 3;
    """
    params = (vec_lit, airline, ticket_class, route_type, group_policy, vec_lit)
    with psycopg2.connect(POLICY_DB) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            raw = cur.fetchall()
    return [
        {
            "policy_type":      r[0],
            "policy_desc":      r[1],
            "condition_decs":   r[2],
            "note":             r[3],
            "applied_pax_type": r[4],
            "score":            float(r[5]),
        }
        for r in raw
    ]


async def query_request_node(state: CombinedState) -> CombinedState:
    slots = state.get("slots") or {}
    question = state.get("question") or ""
    rows = await asyncio.to_thread(_query_policy_sync, slots, question)
    state["retrieved_rows"] = rows
    return state


def _slot_repr(slots: dict, name: str) -> str:
    v = slots.get(name)
    if isinstance(v, list):
        return ", ".join(v) if v else "(không có)"
    return v or "(không có)"


async def generate_request_node(state: CombinedState) -> CombinedState:
    slots = state.get("slots") or {}
    rows = state.get("retrieved_rows") or []

    context_lines = []
    for i, r in enumerate(rows, 1):
        context_lines.append(
            f"[{i}] ({r['policy_type']}) {r['policy_desc']}\n"
            f"    Điều kiện: {r['condition_decs'] or '—'}\n"
            f"    Ghi chú: {r['note'] or '—'}\n"
            f"    Áp dụng: {r['applied_pax_type'] or '—'}\n"
            f"    (similarity={r['score']:.3f})"
        )
    context = "\n".join(context_lines)

    prompt = (
        "Bạn là trợ lý CSKH của hãng hàng không Vietjet. Trả lời người dùng bằng tiếng Việt, "
        "ngắn gọn, đầy đủ ý, dựa duy nhất vào CONTEXT bên dưới, không bịa.\n\n"
        f"CONTEXT (top match theo vector embedding):\n{context}\n\n"
        f"Thông tin slot: hãng={_slot_repr(slots, 'airline')}, "
        f"hạng vé={_slot_repr(slots, 'ticket_class')}, "
        f"chặng={_slot_repr(slots, 'route_type')}, "
        f"nhóm qui định={_slot_repr(slots, 'group_policy')}\n"
        f"Yêu cầu gốc: {state.get('question') or '(không có)'}\n\n"
        "Trả lời:"
    )

    try:
        resp = await _llm(0.0).ainvoke(prompt)
        answer = resp.content if hasattr(resp, "content") else str(resp)
    except Exception:
        answer = "Dựa trên qui định tìm được:\n" + context

    state["answer"] = answer
    state["done"] = True
    return state


async def escalate_node(state: CombinedState) -> CombinedState:
    slots = state.get("slots") or {}
    state["answer"] = (
        "Rất tiếc, hệ thống chưa có thông tin chính xác cho yêu cầu của bạn "
        f"(hãng: {_slot_repr(slots, 'airline')}, hạng vé: {_slot_repr(slots, 'ticket_class')}, "
        f"chặng: {_slot_repr(slots, 'route_type')}, nhóm: {_slot_repr(slots, 'group_policy')}).\n"
        "Mình sẽ chuyển bạn tới nhân viên tư vấn để hỗ trợ trực tiếp. "
        "Vui lòng giữ máy hoặc liên hệ hotline 1900 1886."
    )
    state["escalate"] = True
    state["done"] = True
    return state


async def off_topic_node(state: CombinedState) -> CombinedState:
    user_text = state.get("question") or ""
    prompt = (
        "Bạn là trợ lý CSKH chuyên về nghiệp vụ hàng không (đổi/hoàn vé, hành lý, "
        "hạng vé, qui định bay, giấy tờ bay...). Người dùng vừa gửi nội dung KHÔNG "
        "liên quan. Hãy trả lời NGẮN, LỊCH SỰ, BẰNG TIẾNG VIỆT, mời họ đặt câu hỏi "
        "đúng chủ đề. Không bịa thông tin.\n\n"
        f"Nội dung user: {user_text}\n\n"
        "Trả lời:"
    )
    try:
        resp = await _llm(0.2).ainvoke(prompt)
        answer = resp.content if hasattr(resp, "content") else str(resp)
    except Exception:
        answer = (
            "Xin lỗi, mình chỉ hỗ trợ các thao tác về vé máy bay Vietjet "
            "(đổi vé, hoàn vé, hành lý, nâng hạng, sửa tên,...). "
            "Bạn vui lòng cung cấp lại yêu cầu đúng chủ đề giúp mình nhé."
        )
    state["answer"] = answer
    state["is_off_topic"] = True
    state["done"] = True
    return state


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------
def _route_after_classify(state: CombinedState) -> str:
    return "qna" if state.get("intent") == "question" else "request"


def _route_after_extract(state: CombinedState) -> str:
    if state.get("is_off_topic"):
        return "off_topic"
    if state.get("next_slot") is None:
        return "query_request"
    return "request_slot"


def _route_after_query_request(state: CombinedState) -> str:
    return "generate_request" if (state.get("retrieved_rows") or []) else "escalate"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------
def build_graph(save_image: bool = True):
    g = StateGraph(CombinedState)

    # entry
    g.add_node("classify_intent", classify_intent_node)

    # nhánh QnA
    g.add_node("qna_route", qna_route_wrapper)
    g.add_node("qna_retrieve", qna_retrieve_wrapper)
    g.add_node("qna_grade", qna_grade_wrapper)
    g.add_node("qna_rewrite", qna_rewrite_wrapper)
    g.add_node("qna_generate", qna_generate_wrapper)

    # nhánh request
    g.add_node("extract_entity", extract_entity_node)
    g.add_node("request_slot", request_slot_node)
    g.add_node("query_request", query_request_node)
    g.add_node("generate_request", generate_request_node)
    g.add_node("escalate", escalate_node)
    g.add_node("off_topic", off_topic_node)

    g.add_edge(START, "classify_intent")
    g.add_conditional_edges(
        "classify_intent",
        _route_after_classify,
        {"qna": "qna_route", "request": "extract_entity"},
    )

    # QnA edges
    g.add_edge("qna_route", "qna_retrieve")
    g.add_edge("qna_retrieve", "qna_grade")
    g.add_conditional_edges(
        "qna_grade",
        _after_grade,
        {"generate": "qna_generate", "rewrite": "qna_rewrite"},
    )
    g.add_edge("qna_rewrite", "qna_retrieve")
    g.add_edge("qna_generate", END)

    # Request edges
    g.add_conditional_edges(
        "extract_entity",
        _route_after_extract,
        {
            "off_topic":     "off_topic",
            "request_slot":  "request_slot",
            "query_request": "query_request",
        },
    )
    g.add_edge("request_slot", END)
    g.add_conditional_edges(
        "query_request",
        _route_after_query_request,
        {"generate_request": "generate_request", "escalate": "escalate"},
    )
    g.add_edge("generate_request", END)
    g.add_edge("escalate", END)
    g.add_edge("off_topic", END)

    compiled = g.compile()
    if save_image:
        try:
            img = compiled.get_graph().draw_mermaid_png()
            out_path = Path(__file__).resolve().parent / "graph.png"
            out_path.write_bytes(img)
        except Exception:
            pass
    return compiled


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


# ---------------------------------------------------------------------------
# Convenience entry
# ---------------------------------------------------------------------------
async def ask(message: str, state: Optional[CombinedState] = None) -> CombinedState:
    """One-shot: classify rồi chạy nhánh tương ứng. Với slot-filling đa lượt, dùng FastAPI server."""
    graph = get_graph()
    init: CombinedState = state or {"slots": {}, "attempts": 0, "done": False}
    init["user_input"] = message
    return await graph.ainvoke(init)


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Quy định hoàn vé Vietjet eco nội địa do bão"
    out = asyncio.run(ask(q))
    print("\n=== Question ===")
    print(q)
    print(f"\n=== Intent: {out.get('intent')} ===")
    print("\n=== Answer ===")
    print(out.get("answer") or out.get("slot_question") or "(không có output)")
