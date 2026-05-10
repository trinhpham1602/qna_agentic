"""
review.py — Pipeline 4 bước cho QnA dataset của hãng bay.

Yêu cầu:
1. Đọc dataset/qna_dataset.csv → list[QnAEntity].
2. Tạo bảng + insert theo chunk (chunk_size = 100).
3. Index trên (airline, ticket_class, route_type, group_policy) + IVFFlat trên vector.
4. LangGraph: slot-filling 4 trường, đủ slot → query DB + so embedding → trả lời.

Lưu ý: chỉ review, KHÔNG chạy. Không execute SQL xuống DB.
"""

from __future__ import annotations

import csv
import uuid
from typing import List, Iterable, Optional, TypedDict, Dict

import psycopg2
import uvicorn
from fastapi import FastAPI, HTTPException
from psycopg2.extras import execute_values
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

from config import DB_CONNECTION_STRING, EMBEDDING_MODEL, LLM_MODEL
from entities.QnAEntity import QnAEntity


# ---------------------------------------------------------------------------
# Mapping CSV header (tiếng Việt) → field của QnAEntity
# ---------------------------------------------------------------------------
CSV_TO_FIELD = {
    "Hãng": "airline",
    "Hạng vé": "ticket_class",
    "Loại chặng": "route_type",
    "Nhóm qui định": "group_policy",
    "Loại qui định": "policy_type",
    "Mô tả qui định": "policy_desc",
    "Điều kiện / Thời hạn": "condition_decs",
    "Ghi chú thêm": "note",
    "Đối tượng áp dụng": "applied_pax_type",
}

CSV_PATH = "dataset/qna_dataset.csv"
TABLE_NAME = "qna_policy"
CHUNK_SIZE = 100
EMBEDDING_DIM = 1024  # intfloat/multilingual-e5-large


# ---------------------------------------------------------------------------
# 1) Read CSV → list[QnAEntity]
# ---------------------------------------------------------------------------
def read_qna_dataset(csv_path: str = CSV_PATH) -> List[QnAEntity]:
    """Đọc CSV và trả về list QnAEntity. id sinh tự động từ row index (1-based).
    embedding_vector để rỗng — sẽ tính sau khi build text."""
    entities: List[QnAEntity] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            kwargs = {field: (row.get(col) or "").strip() for col, field in CSV_TO_FIELD.items()}
            entities.append(
                QnAEntity(
                    id=idx,
                    embedding_vector="",
                    **kwargs,
                )
            )
    return entities


# ---------------------------------------------------------------------------
# Build text dùng cho embedding theo đúng template trong prompt
# ---------------------------------------------------------------------------
def build_embedding_text(e: QnAEntity) -> str:
    return (
        f"\n{e.policy_desc} {e.airline} hạng vé {e.ticket_class} {e.route_type}\n"
        f"Qui định {e.policy_desc}\n"
        f"Áp dụng cho {e.applied_pax_type}\n"
    )


# ---------------------------------------------------------------------------
# 2) Create table + insert by chunks of 100
# ---------------------------------------------------------------------------
DDL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id               INTEGER PRIMARY KEY,
    airline          TEXT NOT NULL,
    ticket_class     TEXT NOT NULL,
    route_type       TEXT NOT NULL,
    group_policy     TEXT NOT NULL,
    policy_type      TEXT,
    policy_desc      TEXT,
    condition_decs   TEXT,
    note             TEXT,
    applied_pax_type TEXT,
    embedding        VECTOR({EMBEDDING_DIM})
);
"""

# 3) Index DDL — B-tree trên 4 trường filter + IVFFlat trên vector
INDEX_DDL = f"""
CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_airline       ON {TABLE_NAME}(airline);
CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_ticket_class  ON {TABLE_NAME}(ticket_class);
CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_route_type    ON {TABLE_NAME}(route_type);
CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_group_policy  ON {TABLE_NAME}(group_policy);
-- Composite index hỗ trợ filter cùng lúc 4 slot
CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_slots
    ON {TABLE_NAME}(airline, ticket_class, route_type, group_policy);
-- Vector index cho cosine similarity
CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_embedding
    ON {TABLE_NAME} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""

INSERT_SQL = f"""
INSERT INTO {TABLE_NAME}
    (id, airline, ticket_class, route_type, group_policy,
     policy_type, policy_desc, condition_decs, note, applied_pax_type, embedding)
VALUES %s
ON CONFLICT (id) DO UPDATE SET
    airline          = EXCLUDED.airline,
    ticket_class     = EXCLUDED.ticket_class,
    route_type       = EXCLUDED.route_type,
    group_policy     = EXCLUDED.group_policy,
    policy_type      = EXCLUDED.policy_type,
    policy_desc      = EXCLUDED.policy_desc,
    condition_decs   = EXCLUDED.condition_decs,
    note             = EXCLUDED.note,
    applied_pax_type = EXCLUDED.applied_pax_type,
    embedding        = EXCLUDED.embedding;
"""


def _chunked(seq: List[QnAEntity], size: int) -> Iterable[List[QnAEntity]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _vec_literal(vec: List[float]) -> str:
    """pgvector literal: '[0.1,0.2,...]'"""
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def setup_and_ingest(entities: List[QnAEntity], chunk_size: int = CHUNK_SIZE) -> None:
    """Tạo extension/table/index và insert theo chunk. Embedding tính 1 lần ở batch."""
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    with psycopg2.connect(DB_CONNECTION_STRING) as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)
            cur.execute(INDEX_DDL)
        conn.commit()

        for batch in _chunked(entities, chunk_size):
            texts = [build_embedding_text(e) for e in batch]
            vectors = embedder.embed_documents(texts)
            rows = [
                (
                    e.id,
                    e.airline,
                    e.ticket_class,
                    e.route_type,
                    e.group_policy,
                    e.policy_type,
                    e.policy_desc,
                    e.condition_decs,
                    e.note,
                    e.applied_pax_type,
                    _vec_literal(vec),
                )
                for e, vec in zip(batch, vectors)
            ]
            with conn.cursor() as cur:
                execute_values(cur, INSERT_SQL, rows, template=None, page_size=chunk_size)
            conn.commit()


# ---------------------------------------------------------------------------
# 4) LangGraph: slot-filling 4 trường rồi query + vector similarity
# ---------------------------------------------------------------------------
REQUIRED_SLOTS = ["airline", "ticket_class", "route_type", "group_policy"]

SLOT_PROMPT = {
    "airline":      "Bạn đang hỏi về hãng bay nào? (vd: Vietjet Air, Vietnam Airlines, Bamboo Airways)",
    "ticket_class": "Hạng vé là gì? (vd: Eco, Business, Skyboss)",
    "route_type":   "Loại chặng là gì? (Nội địa / Quốc tế)",
    "group_policy": "Bạn quan tâm nhóm qui định nào? (vd: Đổi vé, Hoàn vé, Hành lý, Đổi tên)",
}


class SlotState(TypedDict, total=False):
    user_input: str         # input mới nhất từ user
    question: str           # câu hỏi gốc (tự do) — dùng để embed
    slots: dict             # đã điền: {slot_name: List[str]} — multi-value
    next_slot: Optional[str]
    slot_question: str      # prompt để hỏi user slot tiếp theo
    retrieved_rows: list    # rows trả về từ vector search
    answer: str             # answer cuối cùng (do generate_node tạo)
    is_off_topic: bool      # True nếu LLM không extract được slot nào (off-topic)
    escalate: bool          # True nếu không tìm thấy nội dung trong DB
    done: bool


# ---------------------------------------------------------------------------
# Helpers cho LLM-based slot extraction
# ---------------------------------------------------------------------------
SYS_PROMPT_EXTRACT_PATH = "sys_prompt/extract_required_slots"

_sys_prompt_cache: Optional[str] = None


def _load_sys_prompt(path: str = SYS_PROMPT_EXTRACT_PATH) -> str:
    """Đọc system prompt từ file (cache module-level)."""
    global _sys_prompt_cache
    if _sys_prompt_cache is None:
        with open(path, "r", encoding="utf-8") as f:
            _sys_prompt_cache = f.read()
    return _sys_prompt_cache


def _parse_slot_json(raw: str) -> Dict[str, List[str]]:
    """Parse JSON output của LLM thành {slot: List[str]}, robust với markdown/junk."""
    import json
    import re

    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        data = json.loads(text)
    except Exception:
        data = {}

    out: Dict[str, List[str]] = {}
    for s in REQUIRED_SLOTS:
        v = data.get(s, [])
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            v = []
        out[s] = [str(x).strip() for x in v if str(x).strip()]
    return out


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


def extract_entity_node(state: SlotState) -> SlotState:
    """
    NODE 1: Dùng LLM (sys_prompt/extract_required_slots) để extract slot
    từ câu chat của user. Output là List[str] cho mỗi slot. Chỉ fill vào
    các slot đang thiếu (giữ nguyên slot đã được điền từ lượt trước).
    """
    user_input = (state.get("user_input") or "").strip()
    slots: Dict[str, List[str]] = dict(state.get("slots") or {})

    if user_input and not state.get("question"):
        state["question"] = user_input

    state["is_off_topic"] = False

    if not user_input:
        state["slots"] = slots
        state["next_slot"] = _missing_slot(slots)
        state["user_input"] = ""
        return state

    sys_prompt = _load_sys_prompt()
    extracted: Dict[str, List[str]] = {s: [] for s in REQUIRED_SLOTS}
    try:
        llm = ChatOllama(model=LLM_MODEL, temperature=0)
        resp = llm.invoke([
            ("system", sys_prompt),
            ("human", user_input),
        ])
        raw = resp.content if hasattr(resp, "content") else str(resp)
        extracted = _parse_slot_json(raw)
    except Exception:
        # LLM lỗi → giữ extracted = empty; flow xuống request_slot
        pass

    pre_empty = all(not _is_slot_filled(slots, s) for s in REQUIRED_SLOTS)
    for s in REQUIRED_SLOTS:
        if not _is_slot_filled(slots, s) and extracted.get(s):
            slots[s] = extracted[s]
    post_empty = all(not _is_slot_filled(slots, s) for s in REQUIRED_SLOTS)

    # Off-topic: không có slot nào được fill từ trước, và LLM cũng không extract được gì
    state["is_off_topic"] = pre_empty and post_empty

    state["slots"] = slots
    state["next_slot"] = _missing_slot(slots)
    state["user_input"] = ""
    return state


def request_slot_node(state: SlotState) -> SlotState:
    """
    NODE 2: Yêu cầu user nhập 1 hoặc nhiều slot còn thiếu.
    """
    slots = state.get("slots") or {}
    missing = [s for s in REQUIRED_SLOTS if not slots.get(s)]

    if not missing:
        state["slot_question"] = ""
        return state

    if len(missing) == 1:
        state["slot_question"] = SLOT_PROMPT[missing[0]]
    else:
        lines = [f"- {SLOT_PROMPT[s]}" for s in missing]
        state["slot_question"] = (
            "Vui lòng cung cấp thêm các thông tin sau (có thể trả lời gộp 1 câu):\n"
            + "\n".join(lines)
        )
    state["done"] = False
    return state


def _pick_one(slots: dict, name: str) -> str:
    """Lấy 1 giá trị đại diện từ slot list (dùng để filter SQL exact)."""
    v = slots.get(name)
    if isinstance(v, list):
        return v[0] if v else ""
    return v or ""


def query_node(state: SlotState) -> SlotState:
    """Đủ 4 slot: filter exact + rank by cosine distance. Lưu rows vào state."""
    slots = state["slots"]
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    airline      = _pick_one(slots, "airline")
    ticket_class = _pick_one(slots, "ticket_class")
    route_type   = _pick_one(slots, "route_type")
    group_policy = _pick_one(slots, "group_policy")

    query_text = state.get("question") or (
        f"{group_policy} {airline} hạng vé {ticket_class} {route_type}"
    )
    qvec = embedder.embed_query(query_text)

    sql = f"""
        SELECT policy_type, policy_desc, condition_decs, note, applied_pax_type,
               1 - (embedding <=> %s::vector) AS score
        FROM {TABLE_NAME}
        WHERE airline = %s
          AND ticket_class = %s
          AND route_type = %s
          AND group_policy = %s
        ORDER BY embedding <=> %s::vector
        LIMIT 3;
    """
    vec_lit = _vec_literal(qvec)
    params = (vec_lit, airline, ticket_class, route_type, group_policy, vec_lit)

    with psycopg2.connect(DB_CONNECTION_STRING) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            raw = cur.fetchall()

    state["retrieved_rows"] = [
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
    return state


def _slot_repr(slots: dict, name: str) -> str:
    """Hiển thị slot list dưới dạng 'a, b' để đưa vào prompt LLM."""
    v = slots.get(name)
    if isinstance(v, list):
        return ", ".join(v) if v else "(không có)"
    return v or "(không có)"


def generate_node(state: SlotState) -> SlotState:
    """
    NODE 3: Generate câu trả lời từ rows tìm được bằng vector embedding.
    Dùng LLM (Ollama) tổng hợp; fallback template nếu LLM lỗi/không có.
    """
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
        "Bạn là trợ lý CSKH của hãng hàng không. Trả lời người dùng bằng tiếng Việt, "
        "ngắn gọn, đầy đủ ý, dựa duy nhất vào CONTEXT bên dưới, không bịa.\n\n"
        f"CONTEXT (top match theo vector embedding):\n{context}\n\n"
        f"Thông tin slot: hãng={_slot_repr(slots, 'airline')}, "
        f"hạng vé={_slot_repr(slots, 'ticket_class')}, "
        f"chặng={_slot_repr(slots, 'route_type')}, "
        f"nhóm qui định={_slot_repr(slots, 'group_policy')}\n"
        f"Câu hỏi gốc: {state.get('question') or '(không có)'}\n\n"
        "Trả lời:"
    )

    try:
        llm = ChatOllama(model=LLM_MODEL, temperature=0)
        resp = llm.invoke(prompt)
        answer = resp.content if hasattr(resp, "content") else str(resp)
    except Exception:
        answer = "Dựa trên qui định tìm được:\n" + context

    state["answer"] = answer
    state["done"] = True
    return state


def escalate_node(state: SlotState) -> SlotState:
    """
    NODE 4: Khi vector search không trả về kết quả nào → chuyển sang
    nhân viên tư vấn. Đặt cờ escalate=True để FE/CSKH biết.
    """
    slots = state.get("slots") or {}
    state["answer"] = (
        "Rất tiếc, hệ thống chưa có thông tin chính xác cho yêu cầu của bạn "
        f"(hãng: {_slot_repr(slots, 'airline')}, hạng vé: {_slot_repr(slots, 'ticket_class')}, "
        f"chặng: {_slot_repr(slots, 'route_type')}, nhóm qui định: {_slot_repr(slots, 'group_policy')}).\n"
        "Tôi sẽ chuyển bạn tới nhân viên tư vấn để được hỗ trợ trực tiếp. "
        "Vui lòng giữ máy hoặc liên hệ hotline CSKH."
    )
    state["escalate"] = True
    state["done"] = True
    return state


def off_topic_node(state: SlotState) -> SlotState:
    """
    NODE 5: Khi LLM không extract được slot nào → câu hỏi không liên quan
    đến nghiệp vụ máy bay. Dùng LLM trả lời lịch sự BẰNG TIẾNG VIỆT,
    nhắc user đặt câu hỏi đúng chủ đề.
    """
    user_text = state.get("question") or ""
    prompt = (
        "Bạn là trợ lý CSKH chuyên về nghiệp vụ hàng không (vé máy bay, đổi/hoàn vé, "
        "hành lý, hạng vé, qui định bay, giấy tờ bay...). Người dùng vừa hỏi một câu "
        "KHÔNG liên quan đến nghiệp vụ máy bay. Hãy trả lời NGẮN GỌN, LỊCH SỰ, "
        "HOÀN TOÀN BẰNG TIẾNG VIỆT. Giải thích rằng bạn chỉ hỗ trợ các vấn đề về "
        "vé máy bay, và mời người dùng đặt lại câu hỏi đúng chủ đề. Không bịa thông "
        "tin, không trả lời câu hỏi off-topic.\n\n"
        f"Câu hỏi của người dùng: {user_text}\n\n"
        "Trả lời:"
    )
    try:
        llm = ChatOllama(model=LLM_MODEL, temperature=0.2)
        resp = llm.invoke(prompt)
        answer = resp.content if hasattr(resp, "content") else str(resp)
    except Exception:
        answer = (
            "Xin lỗi, tôi chỉ hỗ trợ các câu hỏi liên quan đến nghiệp vụ vé máy bay "
            "(đổi vé, hoàn vé, hành lý, qui định bay, giấy tờ bay,...). "
            "Bạn vui lòng đặt câu hỏi đúng chủ đề giúp mình nhé."
        )

    state["answer"] = answer
    state["is_off_topic"] = True
    state["done"] = True
    return state


def _route_after_extract(state: SlotState) -> str:
    if state.get("is_off_topic"):
        return "off_topic"
    if state.get("next_slot") is None:
        return "query"
    return "request_slot"


def _route_after_query(state: SlotState) -> str:
    return "generate" if (state.get("retrieved_rows") or []) else "escalate"


def build_graph():
    g = StateGraph(SlotState)
    g.add_node("extract_entity", extract_entity_node)
    g.add_node("request_slot", request_slot_node)
    g.add_node("query", query_node)
    g.add_node("generate", generate_node)
    g.add_node("escalate", escalate_node)

    g.set_entry_point("extract_entity")
    g.add_conditional_edges(
        "extract_entity",
        _route_after_extract,
        {
            "query":        "query",
            "request_slot": "request_slot",
        },
    )
    g.add_edge("request_slot", END)
    g.add_conditional_edges(
        "query",
        _route_after_query,
        {"generate": "generate", "escalate": "escalate"},
    )
    g.add_edge("generate", END)
    g.add_edge("escalate", END)

    g_compiled = g.compile()
    try:
        img = g_compiled.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(img)
    except Exception:
        pass
    return g_compiled


# ---------------------------------------------------------------------------
# FastAPI server — dùng LangGraph cho slot-filling đa lượt qua HTTP
# ---------------------------------------------------------------------------
app = FastAPI(title="QnA Policy Slot-Filling API", version="1.0.0")

# Compile graph 1 lần khi import module
_graph = build_graph()

# Lưu state theo thread_id (in-memory). Production nên thay bằng checkpointer.
_sessions: Dict[str, SlotState] = {}


class ThreadResponse(BaseModel):
    thread_id: str
    slot_question: str
    missing_slots: List[str]


class ChatRequest(BaseModel):
    thread_id: str
    message: str


class ChatResponse(BaseModel):
    thread_id: str
    done: bool
    slot_question: str = ""
    missing_slots: List[str] = []
    slots: dict = {}
    answer: str = ""
    is_off_topic: bool = False
    escalate: bool = False


class IngestResponse(BaseModel):
    status: str
    rows_ingested: int


def _new_state() -> SlotState:
    return {
        "slots": {},
        "next_slot": None,
        "user_input": "",
        "question": "",
        "slot_question": "",
        "answer": "",
        "done": False,
    }


def _missing_list(slots: dict) -> List[str]:
    return [s for s in REQUIRED_SLOTS if not _is_slot_filled(slots, s)]


@app.post("/thread", response_model=ThreadResponse)
def create_thread() -> ThreadResponse:
    """Khởi tạo thread mới — trả về câu hỏi đầu tiên (yêu cầu user nhập câu hỏi)."""
    tid = str(uuid.uuid4())
    state = _new_state()
    _sessions[tid] = state
    return ThreadResponse(
        thread_id=tid,
        slot_question="Hãy nhập câu hỏi của bạn về qui định vé máy bay.",
        missing_slots=REQUIRED_SLOTS.copy(),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    Mỗi lần gọi: đẩy `message` vào graph.
    - Nếu còn thiếu slot → trả `slot_question` để hỏi tiếp.
    - Khi đủ 4 slot → graph chạy query_node, trả `answer` và `done=True`.
    """
    state = _sessions.get(req.thread_id)
    if state is None:
        raise HTTPException(status_code=404, detail="thread_id không tồn tại")

    if state.get("done"):
        # Đã trả lời xong — gợi ý mở thread mới
        return ChatResponse(
            thread_id=req.thread_id,
            done=True,
            slots=state.get("slots", {}),
            answer=state.get("answer", ""),
            is_off_topic=bool(state.get("is_off_topic")),
            escalate=bool(state.get("escalate")),
        )

    state["user_input"] = (req.message or "").strip()
    new_state: SlotState = _graph.invoke(state)
    _sessions[req.thread_id] = new_state

    return ChatResponse(
        thread_id=req.thread_id,
        done=bool(new_state.get("done")),
        slot_question=new_state.get("slot_question", ""),
        missing_slots=_missing_list(new_state.get("slots") or {}),
        slots=new_state.get("slots", {}),
        answer=new_state.get("answer", ""),
        is_off_topic=bool(new_state.get("is_off_topic")),
        escalate=bool(new_state.get("escalate")),
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    """Endpoint admin: đọc CSV → embed → insert. Chỉ chạy khi cần seed DB."""
    entities = read_qna_dataset()
    setup_and_ingest(entities)
    return IngestResponse(status="ok", rows_ingested=len(entities))


if __name__ == "__main__":
    uvicorn.run("review:app", host="127.0.0.1", port=8001, reload=False)
