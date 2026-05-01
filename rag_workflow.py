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
# Intent → document category mapping (used by retrieve_node)
# Keyed by OLD concept canonicals; new verb canonicals map via VERB_CONCEPT_MAP.
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
    "search_flight":           "Đặt vé",
    "select_seat":             "Dịch vụ bổ sung",
    "check_in":                "Thủ tục",
    "check_policy":            "Chính sách",
}

# ------------------------------------------------------------------
# Verb canonical → internal concept key (for INTENT_SLOTS / INTENT_SUGGESTIONS lookup)
# Context-independent mappings (one-to-one)
# ------------------------------------------------------------------
VERB_CONCEPT_MAP: dict[str, str] = {
    "dat":        "search_flight",
    "huy":        "refund",
    "doi":        "reissue",
    "hoan":       "refund",
    "bao_luu":    "refund",
    "mua_them":   "add_baggage",
    "chon":       "select_seat",
    "dang_ky":    "special_service_request",
    "thanh_toan": "check_payment",
    "check_in":   "check_in",
    "hoi":        "check_policy",      # default; overridden by entity context
    "kiem_tra":   "check_baggage",     # default; overridden by entity context
    "tim_kiem":   "search_flight",
    "bao_cao":    "payment_issue",
    "xac_nhan":   "check_payment",
}

# Context rules for verbs whose concept depends on entity type / id
# Maps verb → {entity_type_or_id: concept}
_VERB_CONTEXT_RULES: dict[str, dict[str, str]] = {
    "kiem_tra": {
        "payment_issue_type": "payment_issue",
        "payment_method":     "check_payment",
        "passenger_type":     "check_special_passenger",
        "reissue_type":       "reissue",
        "giay_to":            "check_document",
        "giao_dich":          "payment_issue",
        "hanh_ly":            "check_baggage",
        "phi":                "check_payment",
        "chinh_sach":         "check_policy",
        "bua_an":             "check_policy",
        "_default":           "check_policy",
    },
    "hoi": {
        "payment_issue_type": "payment_issue",
        "payment_method":     "check_payment",
        "passenger_type":     "check_special_passenger",
        "giay_to":            "check_document",
        "giao_dich":          "check_payment",
        "hanh_ly":            "check_baggage",
        "phi":                "check_payment",
        "chinh_sach":         "check_policy",
        "bua_an":             "check_policy",
        "_default":           "check_policy",
    },
}


def _resolve_verb_to_concept(
    verb: str,
    entity_types: set[str],
    entity_ids: set[str],
) -> str:
    """Map a verb canonical + entity context → internal concept key for slot/suggestion lookup."""
    ctx_rules = _VERB_CONTEXT_RULES.get(verb)
    if ctx_rules:
        # Check entity types first
        for etype in entity_types:
            if etype in ctx_rules:
                return ctx_rules[etype]
        # Then check entity canonical ids
        for eid in entity_ids:
            if eid in ctx_rules:
                return ctx_rules[eid]
        return ctx_rules.get("_default", VERB_CONCEPT_MAP.get(verb, verb))
    return VERB_CONCEPT_MAP.get(verb, verb)

# Fibonacci escalation thresholds
RISK_ESCALATE_HARD = 8   # escalate with strong prompt
RISK_ESCALATE_SOFT = 5   # gentle escalation suggestion

CONFIDENCE_THRESHOLD = 0.50  # below this → "chưa được cung cấp thông tin"
CONTEXT_DOC_LIMIT    = 3     # max docs sent to LLM (speed)
LLM_MAX_TOKENS       = 320   # num_predict cap for Ollama (speed)

# Giá trị mặc định cho slot khi user không đề cập — không hỏi lại
SLOT_DEFAULTS: dict[str, str] = {
    "airline": "vietjet",
}

# Slot definitions per intent: required slots + câu hỏi hỏi user
INTENT_SLOTS: dict[str, dict] = {
    "refund": {
        "required": ["airline", "cabin_class", "route"],
        "questions": {
            "airline":     "Bạn đang hỏi về hãng hàng không nào? (VietJet / Bamboo / Vietnam Airlines)",
            "cabin_class": "Bạn đang dùng hạng vé nào? (Promo / Eco / Deluxe / SkyBoss)",
            "route":       "Chặng bay của bạn là nội địa hay quốc tế?",
        },
    },
    "reissue": {
        "required": ["airline", "reissue_type", "cabin_class", "route"],
        "questions": {
            "airline":      "Bạn đang hỏi về hãng hàng không nào? (VietJet / Bamboo / Vietnam Airlines)",
            "reissue_type": "Bạn muốn thay đổi điều gì trên vé?\n• Đổi giờ/ngày bay\n• Đổi hành trình\n• Đổi tên hành khách",
            "cabin_class":  "Bạn đang dùng hạng vé nào? (Promo / Eco / Deluxe / SkyBoss)",
            "route":        "Chặng bay của bạn là nội địa hay quốc tế?",
        },
    },
    "check_baggage": {
        "required": ["airline", "cabin_class"],
        "questions": {
            "airline":     "Bạn đang hỏi về hãng hàng không nào? (VietJet / Bamboo / Vietnam Airlines)",
            "cabin_class": "Bạn đang dùng hạng vé nào? (Promo / Eco / Deluxe / SkyBoss)",
        },
    },
    "add_baggage": {
        "required": ["airline", "cabin_class", "route"],
        "questions": {
            "airline":     "Bạn đang hỏi về hãng hàng không nào?",
            "cabin_class": "Bạn đang dùng hạng vé nào?",
            "route":       "Chặng bay của bạn là nội địa hay quốc tế?",
        },
    },
    "no_show": {
        "required": ["airline", "cabin_class"],
        "questions": {
            "airline":     "Bạn đang hỏi về hãng hàng không nào?",
            "cabin_class": "Bạn đang dùng hạng vé nào? (Promo / Eco / Deluxe / SkyBoss)",
        },
    },
    "force_majeure": {
        "required": ["airline", "cabin_class"],
        "questions": {
            "airline":     "Bạn đang hỏi về hãng hàng không nào?",
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
    "payment_issue": {
        "required": ["payment_issue_type"],
        "questions": {
            "payment_issue_type": "Bạn gặp vấn đề gì với thanh toán?\n• Lỗi giao dịch / thanh toán thất bại\n• Đã trừ tiền nhưng chưa nhận vé\n• Bị trừ tiền 2 lần",
        },
    },
    # Không cần slot bổ sung
    "check_payment":  {"required": []},
    "check_policy":   {"required": []},
    "search_flight":  {"required": []},
    "select_seat":    {"required": []},
    "check_in":       {"required": []},
}

# entity.type → tên slot
ENTITY_TYPE_TO_SLOT: dict[str, str] = {
    "cabin_class":        "cabin_class",
    "route":              "route",
    "airline":            "airline",
    "passenger_type":     "passenger_type",
    "reissue_type":       "reissue_type",
    "payment_method":     "payment_method",
    "payment_issue_type": "payment_issue_type",
}

# Nhãn tiếng Việt cho từng giá trị slot
SLOT_LABELS: dict[str, dict[str, str]] = {
    "cabin_class":        {"promo": "Promo", "eco": "Eco Standard", "deluxe": "Deluxe", "skyboss": "SkyBoss"},
    "route":              {"domestic": "nội địa", "international": "quốc tế", "australia": "chặng Úc"},
    "airline":            {"vietjet": "VietJet", "bamboo": "Bamboo Airways", "vna": "Vietnam Airlines"},
    "passenger_type":     {"infant": "trẻ sơ sinh", "child": "trẻ em", "pregnant": "phụ nữ mang thai", "unaccompanied_minor": "trẻ đi một mình"},
    "reissue_type":       {"change_time": "đổi giờ/ngày bay", "change_route": "đổi hành trình", "change_name": "đổi tên hành khách"},
    "payment_method":     {"momo": "Ví MoMo", "bank_card": "Thẻ ngân hàng", "vnpay": "VNPay", "bank_transfer": "Chuyển khoản"},
    "payment_issue_type": {"txn_failed": "Lỗi giao dịch", "ticket_not_received": "Chưa nhận vé", "double_charge": "Trừ tiền 2 lần"},
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
        "Phí đổi giờ bay VietJet nội địa là bao nhiêu?",
        "Vé SkyBoss có được miễn phí đổi hành trình không?",
        "VietJet có hỗ trợ đổi tên hành khách trên vé không?",
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
# Graph-path document splitting constants
# ------------------------------------------------------------------

# field trong data.py  →  (metadata key, label hiển thị)
CABIN_MAP: dict[str, tuple[str, str]] = {
    "promo":   ("promo_eco_basic", "Promo"),
    "eco":     ("eco_standard",    "Eco Standard"),
    "deluxe":  ("deluxe",          "Deluxe"),
    "skyboss": ("skyboss",         "SkyBoss"),
}

# Những intent cần tách doc theo cabin_class
CABIN_SPLIT_INTENTS = {
    "check_baggage", "add_baggage", "reissue",
    "refund", "no_show", "force_majeure",
}

# Tình huống → passenger_type (cho check_special_passenger)
SITUATION_PASSENGER_TYPE: dict[str, str] = {
    "Trẻ em dưới 2 tuổi (Infant)": "infant",
    "Trẻ em 2–12 tuổi (Child)":    "child",
    "Phụ nữ mang thai":             "pregnant",
}

# Tình huống → route (cho check_document)
SITUATION_ROUTE: dict[str, str] = {
    "Giấy tờ bay nội địa":  "domestic",
    "Hộ chiếu bay quốc tế": "international",
    "Visa quốc tế":          "international",
}

# Giá trị coi là "không có quy định" → bỏ qua khi tạo doc
_EMPTY_RULE = {"—", "n/a", "n/a (đã có 30kg)", ""}

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
    import psycopg

    # Bước 1: extension + table trong transaction bình thường
    with psycopg.connect(DB_CONNECTION_STRING) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS faq_documents (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                content TEXT,
                metadata JSONB,
                embedding vector(384)
            );
        """)
        conn.commit()

    # Bước 2: HNSW index cần autocommit=True vì pgvector dùng CONCURRENTLY internally
    with psycopg.connect(DB_CONNECTION_STRING, autocommit=True) as conn:
        conn.execute("""
            CREATE INDEX IF NOT EXISTS faq_documents_embedding_idx
                ON faq_documents USING hnsw (embedding vector_cosine_ops);
        """)

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


def _build_path_docs(item: dict) -> list[Document]:
    """
    Tạo documents theo graph path:
    - Intents cần cabin_class  → 4 docs (1 per cabin)
    - check_special_passenger  → 1 doc với passenger_type metadata
    - check_document           → 1 doc với route metadata
    - Còn lại                  → 1 doc tổng hợp
    "Trả lời mẫu" bị loại khỏi nội dung — thuộc system prompt, không nên embed.
    """
    intent    = item.get("intent", "")
    situation = item.get("situation", "")
    notes     = item.get("important_notes", "")

    base_meta = {
        "category":     item.get("category", ""),
        "intent":       intent,
        "airline":      "vietjet",
        "risk":         item.get("risk", 1),
        "escalate_when": item.get("escalate_when", ""),
    }

    docs: list[Document] = []

    if intent in CABIN_SPLIT_INTENTS:
        # ── 4 docs, mỗi doc = 1 path (airline, intent, cabin_class) ──────
        for cabin, (field, label) in CABIN_MAP.items():
            rule = (item.get(field) or "").strip()
            if rule.lower() in _EMPTY_RULE:
                continue
            content = (
                f"Hãng: VietJet | Hạng vé: {label}\n"
                f"Tình huống: {situation}\n"
                f"Quy định: {rule}\n"
                f"Ghi chú: {notes}"
            )
            docs.append(Document(
                page_content=content,
                metadata={**base_meta, "cabin_class": cabin},
            ))

    elif intent in ("check_special_passenger", "special_service_request"):
        # ── 1 doc, thêm passenger_type nếu nhận dạng được ────────────────
        rule = (item.get("promo_eco_basic") or "").strip()
        content = (
            f"Hãng: VietJet\n"
            f"Tình huống: {situation}\n"
            f"Quy định: {rule}\n"
            f"Ghi chú: {notes}"
        )
        meta = {**base_meta}
        pt = SITUATION_PASSENGER_TYPE.get(situation)
        if pt:
            meta["passenger_type"] = pt
        docs.append(Document(page_content=content, metadata=meta))

    elif intent == "check_document":
        # ── 1 doc, thêm route ─────────────────────────────────────────────
        rule = (item.get("promo_eco_basic") or "").strip()
        content = (
            f"Hãng: VietJet\n"
            f"Tình huống: {situation}\n"
            f"Quy định: {rule}\n"
            f"Ghi chú: {notes}"
        )
        meta = {**base_meta}
        rt = SITUATION_ROUTE.get(situation)
        if rt:
            meta["route"] = rt
        docs.append(Document(page_content=content, metadata=meta))

    else:
        # ── 1 doc tổng hợp (check_payment, payment_issue, …) ─────────────
        rules: dict[str, str] = {
            "Promo":   item.get("promo_eco_basic", ""),
            "Eco":     item.get("eco_standard", ""),
            "Deluxe":  item.get("deluxe", ""),
            "SkyBoss": item.get("skyboss", ""),
        }
        unique = list(dict.fromkeys(
            v.strip() for v in rules.values()
            if v and v.strip().lower() not in _EMPTY_RULE
        ))
        rule_text = unique[0] if len(unique) == 1 else " | ".join(
            f"{k}: {v}" for k, v in rules.items()
            if v and v.strip().lower() not in _EMPTY_RULE
        )
        content = (
            f"Hãng: VietJet\n"
            f"Tình huống: {situation}\n"
            f"Quy định: {rule_text}\n"
            f"Ghi chú: {notes}"
        )
        docs.append(Document(page_content=content, metadata=base_meta))

    return docs


def ingest_default_faq(force_reingest: bool = False):
    from data import faq_data

    docs: list[Document] = []
    for item in faq_data:
        docs.extend(_build_path_docs(item))

    _all_documents.extend(docs)
    _rebuild_bm25()

    try:
        already_populated = bool(vector_store.similarity_search("test", k=1))
    except Exception:
        already_populated = False

    if force_reingest or not already_populated:
        vector_store.add_documents(docs)
        logger.info(f"Ingested {len(docs)} path-split docs to vector store")
    else:
        logger.info(f"Vector store already populated — loaded {len(docs)} path-split docs into memory only")


# ------------------------------------------------------------------
# LangGraph state
# ------------------------------------------------------------------

class GraphState(TypedDict):
    question:             str
    entities:             List[dict]   # extracted noun entities
    intents:              List[dict]   # extracted verb intents (each has "concept" resolved)
    intent_mode:          str          # "booking" | "qna" | "ambiguous"
    filled_slots:         dict         # tích luỹ qua nhiều lượt hội thoại
    missing_slots:        List[str]    # slot còn thiếu cho intent hiện tại
    slot_question:        str          # câu hỏi hỏi slot tiếp theo
    reconstructed_query:  str          # câu hoàn chỉnh khi đủ slot
    is_off_topic:         bool         # True khi câu hỏi ngoài nghiệp vụ
    graph_risk:           int          # max Fibonacci risk từ matched intents
    confidence:           float        # 0.0–1.0 từ fuzzy intent match score
    suggested_questions:  List[str]    # pre-computed từ INTENT_SUGGESTIONS
    graph_queries:        List[str]    # queries sinh từ graph traversal
    documents:            List[Document]
    max_risk_level:       int
    final_answer:         str


# ------------------------------------------------------------------
# Knowledge-graph loader (lazy, cached at module level)
# Reads from dataset/claude_dataset (triplet format) via graph_db.
# ------------------------------------------------------------------

_kg_cache: dict = {}


def _load_kg() -> dict:
    """Load claude_dataset once; build KnowledgeGraph + label maps."""
    global _kg_cache
    if _kg_cache:
        return _kg_cache

    from graph_db import KnowledgeGraph

    kg = KnowledgeGraph.from_dataset("dataset/claude_dataset")

    # Build label maps from entities.json and intents.json
    entity_labels: dict[str, str] = {}
    intent_labels: dict[str, str] = {}
    try:
        with open("dataset/entities.json") as f:
            for e in json.load(f):
                entity_labels[e["id"]] = e.get("label", e["id"])
        with open("dataset/intents.json") as f:
            for i in json.load(f):
                intent_labels[i["id"]] = i.get("label", i["id"])
    except Exception as exc:
        logger.warning("Label maps incomplete: %s", exc)

    _kg_cache = {"kg": kg, "entity_labels": entity_labels, "intent_labels": intent_labels}
    return _kg_cache


# Semantic relations that don't correspond to an intent verb
_SEMANTIC_RELATIONS = {"thuoc", "la", "co", "lien_quan", "yeu_cau", "anh_huong"}


def _edge_to_query(from_id: str, relation: str, to_id: str, entity_labels: dict, intent_labels: dict) -> str:
    """Convert a KG edge (triplet) to a Vietnamese retrieval query."""
    from_label = entity_labels.get(from_id, from_id)
    to_label   = entity_labels.get(to_id,   to_id)

    if relation == "yeu_cau":
        return f"VietJet {from_label} yêu cầu {to_label}"
    if relation == "lien_quan":
        return f"VietJet {from_label} liên quan đến {to_label}"
    if relation in ("thuoc", "la"):
        return f"{from_label} là loại {to_label} VietJet"
    if relation == "co":
        return f"VietJet {from_label} có {to_label}"
    if relation in _SEMANTIC_RELATIONS:
        return f"VietJet {from_label} {to_label}"

    # Verb / intent edge
    intent_label = intent_labels.get(relation, relation)
    return f"{intent_label} {to_label} VietJet"


# ------------------------------------------------------------------
# Node: entity & intent extraction
# ------------------------------------------------------------------

async def entity_extraction_node(state: GraphState) -> GraphState:
    logger.info("---ENTITY EXTRACTION NODE---")
    from utils import extract_entities, detect_intent_mode, Entity as EntityModel

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
    prev_missing = state.get("missing_slots", [])
    if not matched_intents and prev_missing:
        matched_intents = list(state.get("intents") or [])
        logger.info("Slot continuation — giữ intents: %s", [i["canonical"] for i in matched_intents])
    # ─────────────────────────────────────────────────────────────────

    # ── Resolve verb → concept for slot/suggestion/category lookup ───
    entity_types = {e["type"] for e in matched_entities}
    entity_ids   = {e["canonical"] for e in matched_entities}
    for intent in matched_intents:
        intent["concept"] = _resolve_verb_to_concept(intent["canonical"], entity_types, entity_ids)
    # ─────────────────────────────────────────────────────────────────

    graph_risk = max((i.get("risk", 1) for i in matched_intents), default=1)
    confidence = max((i.get("best_score", 0) for i in matched_intents), default=0) / 100.0

    # Detect booking vs Q&A mode
    mode_result  = detect_intent_mode(question, matched_intents)
    intent_mode  = mode_result["mode"]

    # Pre-compute suggested questions keyed by concept
    seen: dict[str, None] = {}
    for intent in matched_intents:
        concept = intent.get("concept", intent["canonical"])
        for q in INTENT_SUGGESTIONS.get(concept, []):
            seen[q] = None
    suggested_questions = list(seen)[:3]

    logger.info("Entities   : %s", [e["canonical"] for e in matched_entities])
    logger.info("Intents    : %s", [(i["canonical"], i.get("concept"), i.get("risk")) for i in matched_intents])
    logger.info("Mode       : %s | Confidence: %.2f | Risk: %d", intent_mode, confidence, graph_risk)

    return {
        "entities":            matched_entities,
        "intents":             matched_intents,
        "intent_mode":         intent_mode,
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
    intents  = state.get("intents", [])
    entities = state.get("entities", [])

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

    # ── Phân biệt reissue vs refund (doi vs hoan/huy) khi cả hai match ─
    reissue_i = next((i for i in intents if i.get("concept") == "reissue"), None)
    refund_i  = next((i for i in intents if i.get("concept") == "refund"),  None)
    if reissue_i and refund_i:
        diff = abs(reissue_i.get("best_score", 0) - refund_i.get("best_score", 0))
        if diff < 12:
            return {
                "filled_slots":        state.get("filled_slots") or {},
                "missing_slots":       ["__disambiguate__"],
                "slot_question":       "Bạn muốn **đổi vé** (giữ lại vé, thay đổi thông tin) hay **hoàn vé** (hủy vé, lấy lại tiền)?",
                "reconstructed_query": "",
                "is_off_topic":        False,
            }
    # ─────────────────────────────────────────────────────────────────

    # Merge slot cũ với entity mới
    filled: dict = dict(state.get("filled_slots") or {})
    for entity in entities:
        slot_name = ENTITY_TYPE_TO_SLOT.get(entity.get("type", ""))
        if slot_name:
            filled[slot_name] = entity["canonical"]

    # Áp dụng SLOT_DEFAULTS cho slot chưa có giá trị (không hỏi user)
    for slot, default_val in SLOT_DEFAULTS.items():
        if slot not in filled:
            filled[slot] = default_val
            logger.info("Slot default: %s = %s", slot, default_val)

    # Primary intent = intent có score cao nhất; use concept for slot config lookup
    primary         = max(intents, key=lambda i: i.get("best_score", 0))
    primary_concept = primary.get("concept", primary["canonical"])
    cfg             = INTENT_SLOTS.get(primary_concept, {"required": []})
    required        = cfg.get("required", [])
    missing         = [s for s in required if s not in filled]

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
    # Chỉ dừng lại hỏi khi cần phân biệt reissue vs refund (ambiguous)
    if state.get("missing_slots") == ["__disambiguate__"]:
        return "ask_slot"
    # Câu hỏi chung về quy định → trả lời ngay, slot_question gắn cuối answer
    return "graph_traversal"


# ------------------------------------------------------------------
# Node: knowledge-graph traversal → generate multi-angle queries
# ------------------------------------------------------------------

async def graph_traversal_node(state: GraphState) -> GraphState:
    logger.info("---GRAPH TRAVERSAL NODE---")
    kg_data        = _load_kg()
    kg             = kg_data["kg"]
    entity_labels  = kg_data["entity_labels"]
    intent_labels  = kg_data["intent_labels"]

    entities = state.get("entities", [])
    intents  = state.get("intents",  [])
    question = state["question"]

    # Seed with entity canonicals, intent verb canonicals, and resolved concepts
    seed_ids: set[str] = set()
    for e in entities:
        seed_ids.add(e["canonical"])
    for i in intents:
        seed_ids.add(i["canonical"])
        if i.get("concept"):
            seed_ids.add(i["concept"])

    queries: list[str] = [question]
    seen: set[tuple] = set()

    for seed in seed_ids:
        for edge in kg.neighbors(seed):
            key = (seed, edge["relation"], edge["to"])
            if key in seen:
                continue
            seen.add(key)
            q = _edge_to_query(seed, edge["relation"], edge["to"], entity_labels, intent_labels)
            if q:
                queries.append(q)

            # 2nd hop for semantic-rich relations
            if edge["relation"] in ("yeu_cau", "lien_quan", "co"):
                for edge2 in kg.neighbors(edge["to"]):
                    key2 = (edge["to"], edge2["relation"], edge2["to"])
                    if key2 in seen:
                        continue
                    seen.add(key2)
                    q2 = _edge_to_query(edge["to"], edge2["relation"], edge2["to"], entity_labels, intent_labels)
                    if q2:
                        queries.append(q2)

    # Deduplicate, preserve order, cap at 8 queries (speed)
    seen_q: set[str] = set()
    unique: list[str] = []
    for q in queries:
        if q and q not in seen_q:
            seen_q.add(q)
            unique.append(q)
    unique = unique[:8]

    logger.info("Graph queries (%d):", len(unique))
    for q in unique:
        logger.info("  → %s", q)

    return {"graph_queries": unique}


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
        "Bạn là trợ lý AI thân thiện. Trả lời hoàn toàn bằng tiếng Việt, "
        "giọng gần gũi, dùng 'bạn' và 'mình', thêm 'nhé' khi phù hợp. "
        "Nếu không biết câu trả lời, thành thật nói và gợi ý hướng giải quyết."
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
    question      = state["question"]
    intents       = state.get("intents", [])
    filled_slots  = state.get("filled_slots") or {}
    graph_queries = state.get("graph_queries") or [question]

    # ── Metadata filter từ filled_slots ──────────────────────────────────
    meta_filter: dict = {}
    for slot in ("airline", "cabin_class", "passenger_type", "route"):
        if filled_slots.get(slot):
            meta_filter[slot] = filled_slots[slot]

    logger.info("Slot filter: %s | Graph queries: %d", meta_filter, len(graph_queries))

    # 1. Semantic search — chạy mỗi graph query, merge kết quả
    all_pg_docs: list[Document] = []
    for q in graph_queries:
        try:
            hits = vector_store.similarity_search(
                q, k=VECTOR_K,
                filter=meta_filter if meta_filter else None,
            )
            all_pg_docs.extend(hits)
        except Exception as e:
            logger.error("Vector search failed for query '%s': %s", q, e)

    # Deduplicate by content while preserving retrieval order
    seen_content: set[str] = set()
    pg_docs: list[Document] = []
    for doc in all_pg_docs:
        if doc.page_content not in seen_content:
            seen_content.add(doc.page_content)
            pg_docs.append(doc)

    # Fallback: bỏ filter nếu không tìm thấy gì
    if not pg_docs and meta_filter:
        logger.info("Filter returned 0 results — fallback to unfiltered search")
        for q in graph_queries[:3]:
            try:
                hits = vector_store.similarity_search(q, k=VECTOR_K)
                for doc in hits:
                    if doc.page_content not in seen_content:
                        seen_content.add(doc.page_content)
                        pg_docs.append(doc)
            except Exception as e:
                logger.error("Unfiltered vector search failed: %s", e)

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
    best     = reranked[:CONTEXT_DOC_LIMIT]

    if not best:
        logger.warning("No documents retrieved")
        return {"documents": []}

    # 4. Khi có slot filter chính xác → dùng trực tiếp top docs, không expand
    #    Khi không có filter → expand theo category để LLM có thêm context
    if meta_filter:
        logger.info(f"Slot-filtered retrieval: {len(best)} docs")
        return {"documents": best}

    intent_categories = [
        INTENT_CATEGORY_MAP[concept]
        for i in intents
        if (concept := i.get("concept", i["canonical"])) in INTENT_CATEGORY_MAP
    ]
    top_category = intent_categories[0] if intent_categories else best[0].metadata.get("category")

    if top_category and _all_documents:
        expanded = [d for d in _all_documents if d.metadata.get("category") == top_category]
        logger.info(f"Category expand: {len(expanded)} docs for '{top_category}'")
        return {"documents": expanded[:CONTEXT_DOC_LIMIT]}

    return {"documents": best}


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

    filled_slots = state.get("filled_slots", {})

    slot_ctx = ""
    if filled_slots:
        parts = [labels.get(v, v) for s, labels in SLOT_LABELS.items() if (v := filled_slots.get(s))]
        if parts:
            slot_ctx = "Thông tin đã xác nhận: " + ", ".join(parts) + ".\n"

    sys_prompt = (
        "Bạn là trợ lý CSKH thân thiện, nhiệt tình. Trả lời HOÀN TOÀN bằng tiếng Việt.\n"
        "Giọng điệu ấm áp, gần gũi — dùng 'bạn', xưng 'mình', thêm 'nhé'/'ạ' khi phù hợp.\n"
        "CHỈ dùng thông tin từ tài liệu bên dưới. Nếu không có, nói: 'Thông tin này mình chưa tìm thấy, bạn liên hệ hotline để được hỗ trợ thêm nhé.'\n"
        "Dùng thẻ HTML <b>text</b> để in đậm từ quan trọng. KHÔNG dùng markdown **.\n"
        f"{slot_ctx}"
        f"Tài liệu tham khảo:\n{context}\n"
        "Trả lời ngắn gọn, dễ hiểu."
    )

    try:
        response = await llm.ainvoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=question),
        ])
        answer = response.content
    except Exception as e:
        logger.error(f"LLM error: {e}")
        answer = "Xin lỗi, hệ thống đang quá tải. Vui lòng thử lại sau."

    # Nếu còn slot chưa có → gắn câu hỏi nhẹ cuối answer để làm rõ thêm
    slot_question = state.get("slot_question", "")
    missing       = [s for s in state.get("missing_slots", []) if s != "__disambiguate__"]
    if slot_question and missing:
        answer += f"\n\n💬 <i>Để mình tư vấn chính xác hơn: {slot_question}</i>"

    return {
        "final_answer":        answer,
        "suggested_questions": suggested,
    }


async def escalate_api_action() -> str:
    logger.info("---ESCALATION ACTION---")
    await asyncio.sleep(2)
    return "Hệ thống đã kết nối thành công. Một nhân viên CSKH sẽ tiếp nhận và phản hồi bạn trong giây lát."


# ------------------------------------------------------------------
# Build LangGraph
#
# START
#   → entity_extraction
#   → slot_filling  ──┬── (off_topic)       → off_topic  ──┐
#                     ├── (ask_slot)        → ask_slot   ──┤
#                     └── (graph_traversal) → graph_traversal
#                                               → retrieve
#                                                 → assess_risk
#                                                   → generate ──┘
#                                                        → END
# ------------------------------------------------------------------
builder = StateGraph(GraphState)
builder.add_node("entity_extraction", entity_extraction_node)
builder.add_node("slot_filling",      slot_filling_node)
builder.add_node("ask_slot",          ask_slot_node)
builder.add_node("off_topic",         off_topic_node)
builder.add_node("graph_traversal",   graph_traversal_node)
builder.add_node("retrieve",          retrieve_node)
builder.add_node("assess_risk",       assess_risk_node)
builder.add_node("generate",          generate_node)

builder.add_edge(START,               "entity_extraction")
builder.add_edge("entity_extraction", "slot_filling")
builder.add_conditional_edges(
    "slot_filling",
    _route_after_slots,
    {"off_topic": "off_topic", "ask_slot": "ask_slot", "graph_traversal": "graph_traversal"},
)
builder.add_edge("graph_traversal",  "retrieve")
builder.add_edge("retrieve",         "assess_risk")
builder.add_edge("assess_risk",      "generate")
builder.add_edge("generate",         END)
builder.add_edge("ask_slot",         END)
builder.add_edge("off_topic",        END)
