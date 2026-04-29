import asyncio
import json
from typing import List, TypedDict
import logging

from psycopg_pool import ConnectionPool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from config import (
    DB_CONNECTION_STRING,
    EMBEDDING_MODEL,
    LLM_MODEL,
    COLLECTION_NAME,
    VECTOR_K,
    BM25_K,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Intent → document category mapping
# ------------------------------------------------------------------
INTENT_CATEGORY_MAP: dict[str, str] = {
    "check_baggage":           "Hành lý",
    "add_baggage":             "Hành lý",
    "reissue":                 "Đổi vé",
    "refund":                  "Hoàn vé",
    "no_show":                 "Hoàn vé",
    "force_majeure":           "Hoàn vé",
    "check_special_passenger": "Hành khách đặc biệt",
    "check_document":          "Hành khách đặc biệt",
    "special_service_request": "Hành khách đặc biệt",
    "check_payment":           "Thanh toán & giá vé",
    "payment_issue":           "Thanh toán & giá vé",
}

# Fibonacci escalation thresholds
RISK_ESCALATE_HARD = 8   # escalate with strong prompt
RISK_ESCALATE_SOFT = 5   # gentle escalation suggestion

CONFIDENCE_THRESHOLD = 0.50  # below this → "chưa được cung cấp thông tin"
CONTEXT_DOC_LIMIT    = 3     # max docs sent to LLM (speed)
LLM_MAX_TOKENS       = 320   # num_predict cap for Ollama (speed)

# Slot definitions per intent: required slots + câu hỏi hỏi user
INTENT_SLOTS: dict[str, dict] = {
    "refund": {
        "required": ["cabin_class", "route"],
        "questions": {
            "cabin_class": "Bạn đang dùng hạng vé nào? (Promo / Eco / Deluxe / SkyBoss)",
            "route":       "Chặng bay của bạn là nội địa hay quốc tế?",
        },
    },
    "reissue": {
        "required": ["cabin_class", "route"],
        "questions": {
            "cabin_class": "Bạn đang dùng hạng vé nào? (Promo / Eco / Deluxe / SkyBoss)",
            "route":       "Chặng bay của bạn là nội địa hay quốc tế?",
        },
    },
    "check_baggage": {
        "required": ["cabin_class"],
        "questions": {
            "cabin_class": "Bạn đang dùng hạng vé nào? (Promo / Eco / Deluxe / SkyBoss)",
        },
    },
    "add_baggage": {
        "required": ["cabin_class", "route"],
        "questions": {
            "cabin_class": "Bạn đang dùng hạng vé nào?",
            "route":       "Chặng bay của bạn là nội địa hay quốc tế?",
        },
    },
    "no_show": {
        "required": ["cabin_class"],
        "questions": {
            "cabin_class": "Bạn đang dùng hạng vé nào? (Promo / Eco / Deluxe / SkyBoss)",
        },
    },
    "force_majeure": {
        "required": ["cabin_class"],
        "questions": {
            "cabin_class": "Bạn đang dùng hạng vé nào?",
        },
    },
    "check_special_passenger": {
        "required": ["passenger_type"],
        "questions": {
            "passenger_type": "Bạn đang hỏi về đối tượng hành khách nào?\n• Trẻ sơ sinh (dưới 2 tuổi)\n• Trẻ em (2–12 tuổi)\n• Phụ nữ mang thai\n• Trẻ đi một mình",
        },
    },
    "special_service_request": {
        "required": ["passenger_type"],
        "questions": {
            "passenger_type": "Dịch vụ đặc biệt dành cho đối tượng nào? (Trẻ sơ sinh / Trẻ đi một mình / Phụ nữ mang thai)",
        },
    },
    "check_document": {
        "required": ["route"],
        "questions": {
            "route": "Bạn đang hỏi giấy tờ cho chuyến bay nội địa hay quốc tế?",
        },
    },
    # Không cần slot bổ sung
    "check_payment":  {"required": []},
    "payment_issue":  {"required": []},
    "check_policy":   {"required": []},
    "search_flight":  {"required": []},
    "select_seat":    {"required": []},
}

# entity.type → tên slot
ENTITY_TYPE_TO_SLOT: dict[str, str] = {
    "cabin_class":    "cabin_class",
    "route":          "route",
    "airline":        "airline",
    "passenger_type": "passenger_type",
}

# Nhãn tiếng Việt cho từng giá trị slot
SLOT_LABELS: dict[str, dict[str, str]] = {
    "cabin_class":    {"promo": "Promo", "eco": "Eco Standard", "deluxe": "Deluxe", "skyboss": "SkyBoss"},
    "route":          {"domestic": "nội địa", "international": "quốc tế", "australia": "chặng Úc"},
    "airline":        {"vietjet": "VietJet", "bamboo": "Bamboo Airways", "vna": "Vietnam Airlines"},
    "passenger_type": {"infant": "trẻ sơ sinh", "child": "trẻ em", "pregnant": "phụ nữ mang thai", "unaccompanied_minor": "trẻ đi một mình"},
}

# Pre-computed related questions per intent — zero LLM cost
INTENT_SUGGESTIONS: dict[str, list[str]] = {
    "check_baggage": [
        "Hành lý xách tay VietJet được bao nhiêu kg?",
        "Làm thế nào để mua thêm hành lý ký gửi?",
        "Pin dự phòng có được mang lên máy bay không?",
    ],
    "add_baggage": [
        "Mua thêm hành lý ký gửi ở đâu thì rẻ hơn?",
        "Hành lý ký gửi SkyBoss miễn phí bao nhiêu kg?",
        "Có thể mua thêm hành lý sau khi đã đặt vé không?",
    ],
    "reissue": [
        "Phí đổi vé VietJet nội địa là bao nhiêu?",
        "Đổi tên trên vé có được không?",
        "Phải đổi vé trước giờ bay bao lâu?",
    ],
    "refund": [
        "Vé Promo có hoàn được không?",
        "Phí bảo lưu vé nội địa là bao nhiêu?",
        "Hoàn vé khi hãng hủy chuyến thì được gì?",
    ],
    "no_show": [
        "Lỡ chuyến vé Eco có hoàn được không?",
        "Phí bỏ chỗ khi lỡ chuyến là bao nhiêu?",
        "Deluxe lỡ chuyến xử lý thế nào?",
    ],
    "force_majeure": [
        "Ốm không bay được cần giấy tờ gì để hoàn vé?",
        "Bảo lưu bất khả kháng có tốn phí không?",
        "Thời gian xét duyệt hoàn bất khả kháng mất bao lâu?",
    ],
    "check_special_passenger": [
        "Trẻ em dưới 2 tuổi bay có cần vé riêng không?",
        "Phụ nữ mang thai bao nhiêu tuần thì không được bay?",
        "Trẻ em đi một mình có được không?",
    ],
    "check_document": [
        "Bay nội địa cần giấy tờ gì?",
        "Hộ chiếu cần còn hạn bao lâu khi bay quốc tế?",
        "Người Việt đi Thái Lan có cần visa không?",
    ],
    "check_payment": [
        "MoMo hỗ trợ những hình thức thanh toán nào?",
        "Mua vé sớm có rẻ hơn không?",
        "Mua vé cho người khác có được không?",
    ],
    "payment_issue": [
        "Bị trừ tiền nhưng chưa nhận vé phải làm gì?",
        "Hotline hỗ trợ giao dịch VietJet là bao nhiêu?",
        "Thời gian hoàn tiền lỗi giao dịch mất bao lâu?",
    ],
    "search_flight": [
        "Vé VietJet mua sớm bao lâu thì rẻ nhất?",
        "Vé khứ hồi hay một chiều rẻ hơn?",
        "Có những hạng vé nào trên VietJet?",
    ],
    "check_policy": [
        "Chính sách hoàn vé VietJet như thế nào?",
        "Quy định đổi vé VietJet là gì?",
        "Hạng vé SkyBoss có những ưu đãi gì?",
    ],
    "special_service_request": [
        "Đăng ký dịch vụ đặc biệt ở đâu?",
        "Xe lăn có được hỗ trợ miễn phí không?",
        "Trẻ đi một mình cần đăng ký dịch vụ gì?",
    ],
    "select_seat": [
        "Chọn ghế trước trên VietJet có mất phí không?",
        "SkyBoss có được ưu tiên chọn ghế không?",
        "Ghế cạnh cửa sổ có thể chọn trước không?",
    ],
}

# ------------------------------------------------------------------
# Retrieval state (module-level, shared across requests)
# ------------------------------------------------------------------
_all_documents: List[Document] = []
_bm25_retriever: BM25Retriever | None = None

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=DB_CONNECTION_STRING,
    use_jsonb=True,
)


# ------------------------------------------------------------------
# Initialization helpers
# ------------------------------------------------------------------

def init_postgres_db():
    sql = """
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS faq_documents (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        content TEXT,
        metadata JSONB,
        embedding vector(384)
    );
    CREATE INDEX IF NOT EXISTS faq_documents_embedding_idx
        ON faq_documents USING hnsw (embedding vector_cosine_ops);
    """
    with ConnectionPool(DB_CONNECTION_STRING) as pool:
        with pool.connection() as conn:
            conn.execute(sql)
    logger.info("PostgreSQL initialized")


def _rebuild_bm25():
    global _bm25_retriever
    if _all_documents:
        _bm25_retriever = BM25Retriever.from_documents(_all_documents)
        _bm25_retriever.k = BM25_K


def add_documents(docs: List[Document], persist: bool = True):
    if not docs:
        return
    _all_documents.extend(docs)
    _rebuild_bm25()
    if persist:
        vector_store.add_documents(docs)
    logger.info(f"add_documents: +{len(docs)} (persist={persist}), total={len(_all_documents)}")


def get_stats() -> dict:
    from collections import Counter
    cats = Counter(doc.metadata.get("category", "unknown") for doc in _all_documents)
    return {"total": len(_all_documents), "by_category": dict(cats)}


def ingest_default_faq(force_reingest: bool = False):
    from data import faq_data

    docs = []
    for item in faq_data:
        content = (
            f"Tình huống: {item.get('situation')}\n"
            f"Quy định Eco: {item.get('eco_standard')}\n"
            f"Quy định Promo: {item.get('promo_eco_basic')}\n"
            f"Quy định Deluxe: {item.get('deluxe')}\n"
            f"Quy định SkyBoss: {item.get('skyboss')}\n"
            f"Ghi chú: {item.get('important_notes')}\n"
            f"Trả lời mẫu: {item.get('short_answer')}"
        )
        docs.append(Document(
            page_content=content,
            metadata={
                "category":     item.get("category"),
                "intent":       item.get("intent", ""),
                "risk":         item.get("risk", 1),
                "escalate_when": item.get("escalate_when", ""),
            },
        ))

    _all_documents.extend(docs)
    _rebuild_bm25()

    try:
        already_populated = bool(vector_store.similarity_search("test", k=1))
    except Exception:
        already_populated = False

    if force_reingest or not already_populated:
        vector_store.add_documents(docs)
        logger.info(f"Ingested {len(docs)} FAQ docs to vector store")
    else:
        logger.info(f"Vector store already populated — loaded {len(docs)} docs into memory only")


# ------------------------------------------------------------------
# LangGraph state
# ------------------------------------------------------------------

class GraphState(TypedDict):
    question:             str
    entities:             List[dict]   # extracted airline / cabin / route / service / passenger
    intents:              List[dict]   # extracted intents with Fibonacci risk
    filled_slots:         dict         # tích luỹ qua nhiều lượt hội thoại
    missing_slots:        List[str]    # slot còn thiếu cho intent hiện tại
    slot_question:        str          # câu hỏi hỏi slot tiếp theo
    reconstructed_query:  str          # câu hoàn chỉnh khi đủ slot
    is_off_topic:         bool         # True khi câu hỏi ngoài nghiệp vụ
    graph_risk:           int          # max Fibonacci risk từ matched intents
    confidence:           float        # 0.0–1.0 từ fuzzy intent match score
    suggested_questions:  List[str]    # pre-computed từ INTENT_SUGGESTIONS
    documents:            List[Document]
    max_risk_level:       int
    final_answer:         str
    conversation_summary: str


# ------------------------------------------------------------------
# Node: entity & intent extraction
# ------------------------------------------------------------------

async def entity_extraction_node(state: GraphState) -> GraphState:
    logger.info("---ENTITY EXTRACTION NODE---")
    from utils import extract_entities, Entity as EntityModel

    question = state["question"]

    with open("dataset/entities.json") as f:
        raw_entities = json.load(f)
    with open("dataset/intents.json") as f:
        raw_intents = json.load(f)

    entity_models = [EntityModel(**e) for e in raw_entities]
    intent_models = [EntityModel(**i) for i in raw_intents]

    matched_entities = extract_entities(question, entity_models, threshold=55)
    matched_intents  = extract_entities(question, intent_models,  threshold=55)

    # ── Slot continuation ────────────────────────────────────────────
    # Nếu user đang trả lời câu hỏi slot (prev missing_slots != [])
    # mà không gõ lại intent → giữ nguyên intents từ lượt trước
    # để slot_filling_node không nhầm thành off-topic.
    prev_missing = state.get("missing_slots", [])
    if not matched_intents and prev_missing:
        matched_intents = list(state.get("intents") or [])
        logger.info("Slot continuation — giữ intents: %s", [i["canonical"] for i in matched_intents])
    # ─────────────────────────────────────────────────────────────────

    graph_risk = max((i.get("risk", 1) for i in matched_intents), default=1)

    # Confidence = best fuzzy score across matched intents, normalised to 0–1
    confidence = max((i.get("best_score", 0) for i in matched_intents), default=0) / 100.0

    # Pre-compute suggested questions (no LLM, zero latency)
    seen: dict[str, None] = {}
    for intent in matched_intents:
        for q in INTENT_SUGGESTIONS.get(intent["canonical"], []):
            seen[q] = None
    suggested_questions = list(seen)[:3]

    logger.info("Entities   : %s", [e["canonical"] for e in matched_entities])
    logger.info("Intents    : %s", [(i["canonical"], i.get("risk")) for i in matched_intents])
    logger.info("Confidence : %.2f  |  Graph risk: %d", confidence, graph_risk)

    return {
        "entities":            matched_entities,
        "intents":             matched_intents,
        "graph_risk":          graph_risk,
        "confidence":          confidence,
        "suggested_questions": suggested_questions,
    }


# ------------------------------------------------------------------
# Helper: build reconstructed query từ intents + filled_slots
# ------------------------------------------------------------------

def _build_reconstructed_query(intents: list[dict], filled_slots: dict) -> str:
    intent_labels = [i.get("label") or i["canonical"] for i in intents]
    parts = ["Bạn đang hỏi về: " + ", ".join(intent_labels)]

    for slot, values in SLOT_LABELS.items():
        val = filled_slots.get(slot)
        if val:
            parts.append(values.get(val, val))

    return " — ".join(parts) + "."


# ------------------------------------------------------------------
# Node: slot filling — merge slots, detect missing, detect off-topic
# ------------------------------------------------------------------

async def slot_filling_node(state: GraphState) -> GraphState:
    logger.info("---SLOT FILLING NODE---")
    intents    = state.get("intents", [])
    entities   = state.get("entities", [])
    confidence = state.get("confidence", 0.0)

    # Off-topic: không match được intent nào
    if not intents:
        logger.info("Off-topic: không nhận dạng được intent")
        return {
            "is_off_topic":        True,
            "missing_slots":       [],
            "slot_question":       "",
            "reconstructed_query": "",
            "filled_slots":        state.get("filled_slots") or {},
        }

    # Merge slot cũ (từ lượt trước) với entity mới nhận ra trong lượt này
    filled: dict = dict(state.get("filled_slots") or {})
    for entity in entities:
        slot_name = ENTITY_TYPE_TO_SLOT.get(entity.get("type", ""))
        if slot_name:
            filled[slot_name] = entity["canonical"]

    # Primary intent = intent có score cao nhất
    primary = max(intents, key=lambda i: i.get("best_score", 0))
    cfg      = INTENT_SLOTS.get(primary["canonical"], {"required": []})
    required = cfg.get("required", [])
    missing  = [s for s in required if s not in filled]

    slot_question = ""
    if missing:
        slot_question = cfg.get("questions", {}).get(
            missing[0], f"Bạn có thể cho biết thêm về {missing[0]} không?"
        )

    reconstructed = _build_reconstructed_query(intents, filled) if not missing else ""

    logger.info("Slots đã có: %s | Còn thiếu: %s", list(filled.keys()), missing)

    return {
        "filled_slots":        filled,
        "missing_slots":       missing,
        "slot_question":       slot_question,
        "reconstructed_query": reconstructed,
        "is_off_topic":        False,
    }


def _route_after_slots(state: GraphState) -> str:
    if state.get("is_off_topic"):
        return "off_topic"
    if state.get("missing_slots"):
        return "ask_slot"
    return "retrieve"


# ------------------------------------------------------------------
# Node: ask_slot — trả về câu hỏi slot, không gọi LLM / retrieval
# ------------------------------------------------------------------

async def ask_slot_node(state: GraphState) -> GraphState:
    logger.info("---ASK SLOT NODE---")
    question = state.get("slot_question") or "Bạn có thể cung cấp thêm thông tin không?"
    return {"final_answer": question}


# ------------------------------------------------------------------
# Node: off_topic — LLM thuần, không có RAG context
# ------------------------------------------------------------------

async def off_topic_node(state: GraphState) -> GraphState:
    logger.info("---OFF TOPIC NODE---")
    llm = ChatOllama(model=LLM_MODEL, temperature=0.3, num_predict=LLM_MAX_TOKENS)
    sys_prompt = (
        "Bạn là trợ lý AI thông minh. "
        "Hãy trả lời câu hỏi của người dùng hoàn toàn bằng tiếng Việt, "
        "ngắn gọn và hữu ích. "
        "Nếu không biết câu trả lời, hãy nói thật."
    )
    try:
        response = await llm.ainvoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=state["question"]),
        ])
        answer = response.content
    except Exception as e:
        logger.error(f"Off-topic LLM error: {e}")
        answer = "Xin lỗi, mình chưa thể trả lời câu hỏi này lúc này."
    return {"final_answer": answer}


# ------------------------------------------------------------------
# Node: retrieval (intent-aware)
# ------------------------------------------------------------------

async def retrieve_node(state: GraphState) -> GraphState:
    logger.info("---RETRIEVAL NODE---")
    question = state["question"]
    intents  = state.get("intents", [])

    # 1. Semantic search
    try:
        pg_docs = vector_store.similarity_search(question, k=VECTOR_K)
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        pg_docs = []

    # 2. BM25
    bm25_docs = _bm25_retriever.invoke(question) if _bm25_retriever else []

    # 3. Reciprocal Rank Fusion
    scores:  dict[str, float]    = {}
    doc_map: dict[str, Document] = {}
    for doc_list in [pg_docs, bm25_docs]:
        for rank, doc in enumerate(doc_list):
            key = doc.page_content
            scores[key]  = scores.get(key, 0.0) + 1.0 / (rank + 60)
            doc_map[key] = doc

    reranked = [doc_map[k] for k, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    best = reranked[:2]

    if not best:
        logger.warning("No documents retrieved")
        return {"documents": []}

    # 4. Category: prefer intent-derived, fallback to top-retrieved
    intent_categories = [
        INTENT_CATEGORY_MAP[i["canonical"]]
        for i in intents
        if i["canonical"] in INTENT_CATEGORY_MAP
    ]

    if intent_categories:
        top_category = intent_categories[0]
        logger.info(f"Category from intent graph: {top_category!r}")
    else:
        top_category = best[0].metadata.get("category")
        logger.info(f"Category from retrieval fallback: {top_category!r}")

    if top_category and _all_documents:
        expanded = [d for d in _all_documents if d.metadata.get("category") == top_category]
    else:
        expanded = best

    logger.info(f"Expanded to {len(expanded)} docs for category '{top_category}'")
    return {"documents": expanded}


# ------------------------------------------------------------------
# Node: risk assessment (graph risk + document risk)
# ------------------------------------------------------------------

async def assess_risk_node(state: GraphState) -> GraphState:
    logger.info("---ASSESS RISK NODE---")
    graph_risk = state.get("graph_risk", 1)
    doc_risk   = max(
        (doc.metadata.get("risk", 0) for doc in state.get("documents", [])),
        default=0,
    )
    max_risk = max(graph_risk, doc_risk)
    logger.info(f"Risk — graph: {graph_risk}, doc: {doc_risk}, final (Fibonacci): {max_risk}")
    return {"max_risk_level": max_risk}


# ------------------------------------------------------------------
# Node: generate answer
# ------------------------------------------------------------------

async def generate_node(state: GraphState) -> GraphState:
    logger.info("---GENERATE NODE---")
    question   = state["question"]
    docs       = state.get("documents", [])
    max_risk   = state.get("max_risk_level", 1)
    summary    = state.get("conversation_summary", "")
    entities   = state.get("entities", [])
    intents    = state.get("intents", [])
    confidence = state.get("confidence", 0.0)
    suggested  = state.get("suggested_questions", [])

    # Low confidence or no retrieval → skip LLM entirely
    if confidence < CONFIDENCE_THRESHOLD or not docs:
        logger.info("Low confidence (%.2f) — returning 'not provided' response", confidence)
        return {
            "final_answer":        "Xin lỗi, câu hỏi này chưa được cung cấp thông tin trong hệ thống. Vui lòng liên hệ hotline VietJet để được hỗ trợ trực tiếp.",
            "suggested_questions": suggested,
        }

    # Limit context for speed — use short_answer field if present
    top_docs = docs[:CONTEXT_DOC_LIMIT]
    context  = "\n---\n".join(doc.page_content for doc in top_docs)

    llm = ChatOllama(model=LLM_MODEL, temperature=0, num_predict=LLM_MAX_TOKENS)

    # Compact system prompt — luôn trả lời bằng tiếng Việt
    risk_note = ""
    if max_risk >= RISK_ESCALATE_HARD:
        risk_note = " ⚠️ Rủi ro cao — BẮT BUỘC nhắc bấm 'Xác nhận Chuyển nhân viên CSKH'."
    elif max_risk >= RISK_ESCALATE_SOFT:
        risk_note = " Gợi ý nhẹ chuyển nhân viên nếu cần."

    reconstructed = state.get("reconstructed_query", "")
    filled_slots  = state.get("filled_slots", {})

    slot_ctx = ""
    if filled_slots:
        parts = []
        for slot, labels in SLOT_LABELS.items():
            val = filled_slots.get(slot)
            if val:
                parts.append(f"{slot}: {labels.get(val, val)}")
        if parts:
            slot_ctx = "Thông tin đã xác nhận — " + ", ".join(parts) + ".\n"

    sys_prompt = (
        "Bạn là trợ lý CSKH. Trả lời HOÀN TOÀN bằng tiếng Việt, ngắn gọn, rõ ràng.\n"
        f"{slot_ctx}"
        f"Quy định liên quan:\n{context}\n"
    )
    if summary:
        sys_prompt += f"Lịch sử hội thoại: {summary}\n"
    sys_prompt += f"Trả lời trực tiếp.{risk_note}"

    try:
        response = await llm.ainvoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=question),
        ])
        answer = response.content
    except Exception as e:
        logger.error(f"LLM error: {e}")
        answer = "Xin lỗi, hệ thống đang quá tải. Vui lòng thử lại sau."

    # Prepend reconstructed query để user thấy hệ thống đã hiểu đúng câu hỏi
    if reconstructed:
        answer = f"📋 {reconstructed}\n\n{answer}"

    return {
        "final_answer":        answer,
        "suggested_questions": suggested,
    }


# ------------------------------------------------------------------
# Node: conversation memory summary
# ------------------------------------------------------------------

async def summarize_memory_node(state: GraphState) -> GraphState:
    logger.info("---SUMMARIZE MEMORY NODE---")
    old_summary = state.get("conversation_summary", "")
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    prompt = (
        "Viết tóm tắt ngắn bằng tiếng Việt (dưới 50 từ) về bối cảnh đang thảo luận.\n"
        f"Tóm tắt cũ: {old_summary}\n"
        f"Người dùng hỏi: {state.get('question', '')}\n"
        f"Bot đáp: {state.get('final_answer', '')}\n"
        "Tóm tắt mới:"
    )

    try:
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        new_summary = response.content
    except Exception as e:
        logger.error(f"Summarize error: {e}")
        new_summary = old_summary

    return {"conversation_summary": new_summary}


async def escalate_api_action() -> str:
    logger.info("---ESCALATION ACTION---")
    await asyncio.sleep(2)
    return "Hệ thống đã kết nối thành công. Một nhân viên CSKH sẽ tiếp nhận và phản hồi bạn trong giây lát."


# ------------------------------------------------------------------
# Build LangGraph
#
# START
#   → entity_extraction
#   → slot_filling  ──┬── (off_topic)  → off_topic_node  ──┐
#                     ├── (ask_slot)   → ask_slot_node   ──┤
#                     └── (retrieve)   → retrieve         │
#                                          → assess_risk   │
#                                          → generate      │
#                                               └──────────┤
#                                             summarize_memory
#                                                  → END
# ------------------------------------------------------------------
builder = StateGraph(GraphState)
builder.add_node("entity_extraction", entity_extraction_node)
builder.add_node("slot_filling",      slot_filling_node)
builder.add_node("ask_slot",          ask_slot_node)
builder.add_node("off_topic",         off_topic_node)
builder.add_node("retrieve",          retrieve_node)
builder.add_node("assess_risk",       assess_risk_node)
builder.add_node("generate",          generate_node)
builder.add_node("summarize_memory",  summarize_memory_node)

builder.add_edge(START,               "entity_extraction")
builder.add_edge("entity_extraction", "slot_filling")
builder.add_conditional_edges(
    "slot_filling",
    _route_after_slots,
    {"off_topic": "off_topic", "ask_slot": "ask_slot", "retrieve": "retrieve"},
)
builder.add_edge("retrieve",         "assess_risk")
builder.add_edge("assess_risk",      "generate")
builder.add_edge("generate",         "summarize_memory")
builder.add_edge("ask_slot",         "summarize_memory")
builder.add_edge("off_topic",        "summarize_memory")
builder.add_edge("summarize_memory", END)
