import asyncio
import json
import os
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
from utils import predict_type_of_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# FAQ canonical lookup — loaded once from faq_data.json
# ------------------------------------------------------------------

_FAQ_CANONICALS: set[str] = set()


def _load_faq_canonicals() -> None:
    global _FAQ_CANONICALS
    try:
        with open("faq_data.json", "r") as f:
            data = json.load(f)
        _FAQ_CANONICALS = {item["canonical"] for item in data if item.get("canonical")}
        logger.info("FAQ canonicals loaded: %d entries", len(_FAQ_CANONICALS))
    except Exception as exc:
        logger.warning("Could not load faq_data.json for canonical lookup: %s", exc)


_load_faq_canonicals()

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
    "upgrade_class":           "Dịch vụ bổ sung",
    "check_booking":           "Đặt vé",
    "print_ticket":            "Thủ tục",
    "apply_promo":             "Thanh toán & giá vé",
    "buy_insurance":           "Dịch vụ bổ sung",
}

# ------------------------------------------------------------------
# Verb canonical → internal concept key (for INTENT_SLOTS / INTENT_SUGGESTIONS lookup)
# Context-independent mappings (one-to-one)
# ------------------------------------------------------------------
VERB_CONCEPT_MAP: dict[str, str] = {
    "dat":           "search_flight",
    "huy":           "refund",
    "doi":           "reissue",
    "hoan":          "refund",
    "bao_luu":       "refund",
    "mua_them":      "add_baggage",
    "chon":          "select_seat",
    "dang_ky":       "special_service_request",
    "thanh_toan":    "check_payment",
    "check_in":      "check_in",
    "hoi":           "check_policy",      # default; overridden by entity context
    "kiem_tra":      "check_baggage",     # default; overridden by entity context
    "tim_kiem":      "search_flight",
    "bao_cao":       "payment_issue",
    "xac_nhan":      "check_payment",
    "nang_hang":     "upgrade_class",
    "xem_lich":      "check_booking",
    "in_ve":         "print_ticket",
    "ap_ma":         "apply_promo",
    "mua_bao_hiem":  "buy_insurance",
}

# Context rules for verbs whose concept depends on entity type / id
# Maps verb → {entity_type_or_id: concept}
_VERB_CONTEXT_RULES: dict[str, dict[str, str]] = {
    "kiem_tra": {
        "payment_issue_type": "payment_issue",
        "payment_method":     "check_payment",
        "passenger_type":     "check_special_passenger",
        "reissue_type":       "reissue",
        "document_type":      "check_document",
        "baggage_type":       "check_baggage",
        "restricted_item":    "check_baggage",
        "fee_type":           "check_payment",
        "seat_type":          "select_seat",
        "service_type":       "special_service_request",
        "insurance_type":     "buy_insurance",
        "bua_an":             "check_policy",
        "_default":           "check_policy",
    },
    "hoi": {
        "payment_issue_type": "payment_issue",
        "payment_method":     "check_payment",
        "passenger_type":     "check_special_passenger",
        "document_type":      "check_document",
        "baggage_type":       "check_baggage",
        "restricted_item":    "check_baggage",
        "fee_type":           "check_payment",
        "seat_type":          "select_seat",
        "service_type":       "special_service_request",
        "insurance_type":     "buy_insurance",
        "bua_an":             "check_policy",
        "_default":           "check_policy",
    },
    "mua_them": {
        "baggage_type":   "add_baggage",
        "insurance_type": "buy_insurance",
        "_default":       "add_baggage",
    },
    "chon": {
        "seat_type": "select_seat",
        "_default":  "select_seat",
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
        "required": ["airline", "baggage_type", "cabin_class", "route"],
        "questions": {
            "airline":      "Bạn đang hỏi về hãng hàng không nào?",
            "baggage_type": "Bạn muốn mua thêm hành lý xách tay hay ký gửi?",
            "cabin_class":  "Bạn đang dùng hạng vé nào?",
            "route":        "Chặng bay của bạn là nội địa hay quốc tế?",
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
    "select_seat": {
        "required": ["seat_type"],
        "questions": {
            "seat_type": "Bạn muốn chọn loại ghế nào?\n• Ghế cửa sổ\n• Ghế lối đi\n• Ghế hàng đầu\n• Ghế thoát hiểm",
        },
    },
    "upgrade_class": {
        "required": ["airline", "cabin_class"],
        "questions": {
            "airline":     "Bạn muốn nâng hạng vé của hãng nào?",
            "cabin_class": "Bạn muốn nâng lên hạng nào? (Eco / Deluxe / SkyBoss)",
        },
    },
    # Không cần slot bổ sung
    "check_payment":  {"required": []},
    "check_policy":   {"required": []},
    "search_flight":  {"required": []},
    "check_in":       {"required": []},
    "check_booking":  {"required": []},
    "print_ticket":   {"required": []},
    "apply_promo":    {"required": []},
    "buy_insurance":  {"required": []},
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
    "baggage_type":       "baggage_type",
    "seat_type":          "seat_type",
    "document_type":      "document_type",
    "fee_type":           "fee_type",
    "service_type":       "service_type",
    "insurance_type":     "insurance_type",
    "promo_type":         "promo_type",
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
    "baggage_type":       {"xach_tay": "Hành lý xách tay", "ky_gui": "Hành lý ký gửi"},
    "seat_type":          {"ghe_cua_so": "Ghế cửa sổ", "ghe_loi_di": "Ghế lối đi", "ghe_hang_dau": "Ghế hàng đầu", "ghe_thoat_hiem": "Ghế thoát hiểm"},
    "document_type":      {"cmnd_cccd": "CMND/CCCD", "ho_chieu": "Hộ chiếu", "visa": "Visa", "giay_khai_sinh": "Giấy khai sinh", "giay_bac_si": "Giấy bác sĩ"},
    "fee_type":           {"phi_doi_ve": "Phí đổi vé", "phi_hoan_ve": "Phí hoàn vé", "phi_no_show": "Phí bỏ chỗ", "phi_qua_can": "Phí quá cân", "phi_nang_hang": "Phí nâng hạng"},
    "service_type":       {"dich_vu_xe_lan": "Dịch vụ xe lăn", "dich_vu_um": "Dịch vụ trẻ đơn thân"},
    "insurance_type":     {"bao_hiem_du_lich": "Bảo hiểm du lịch"},
    "promo_type":         {"ma_khuyen_mai": "Mã khuyến mãi"},
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
    "upgrade_class": [
        "Nâng hạng vé VietJet lên SkyBoss mất bao nhiêu?",
        "Có thể nâng hạng sau khi đã đặt vé không?",
        "Vé Promo có nâng hạng lên Deluxe được không?",
    ],
    "check_booking": [
        "Xem lịch đặt vé trên app VietJet ở đâu?",
        "Mã đặt chỗ (booking code) là gì?",
        "Đặt vé nhưng không nhận được email xác nhận thì làm sao?",
    ],
    "print_ticket": [
        "Tải vé điện tử VietJet ở đâu?",
        "Vé PDF có dùng được để check-in không?",
        "Gửi lại vé qua email thì làm thế nào?",
    ],
    "apply_promo": [
        "Mã khuyến mãi VietJet nhập ở bước nào?",
        "Mã giảm giá có áp dụng cho tất cả hạng vé không?",
        "Mã voucher có hết hạn không?",
    ],
    "buy_insurance": [
        "Bảo hiểm du lịch VietJet bao gồm những gì?",
        "Mua bảo hiểm chuyến bay thêm bao nhiêu tiền?",
        "Bảo hiểm có bồi thường khi hủy chuyến không?",
    ],
}

# ------------------------------------------------------------------
# Graph-path document splitting constants
# ------------------------------------------------------------------

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
    cabin_classes = item.get("cabin_classes", [])
    docs: list[Document] = []
    base_meta = {
        "airline":      "vietjet",
        "risk":         item.get("risk", 1),
        "escalate_when": item.get("escalate_when", ""),
    }
    for e in cabin_classes:
        for q in item.get("questions", []):
            content = f"""Câu hỏi tình huống: {q} {e.get("name", "")}.
            \nTrả lời: {e.get("content", "")}"""
        docs.append(Document(page_content=content, metadata=base_meta))
  
    return docs

def ingest_default_faq(force_reingest: bool = False):
    docs: list[Document] = []
    with open("faq_data.json", "r") as f:
        faq_data = json.load(f)
        for item in faq_data:
            docs.extend(_build_path_docs(item))

        _all_documents.extend(docs)
        _rebuild_bm25()

        if force_reingest:
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
    # --- new fields for query-type routing ---
    query_type:            str          # "command" | "question"
    faq_canonical_matched: str          # canonical từ faq_data khớp với câu hỏi
    faq_original_question: str          # câu hỏi gốc của topic hiện tại (multi-turn)


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
# ML extractor cache — loaded once, reused across requests
# ------------------------------------------------------------------

_entity_extractor = None
_intent_extractor = None

_ENTITY_MODEL_PATH = "models/entity_extractor.pkl"
_INTENT_MODEL_PATH = "models/intent_extractor.pkl"


def _get_extractors():
    """Load (or auto-train) the ML extractors, cached at module level."""
    global _entity_extractor, _intent_extractor
    if _entity_extractor is not None and _intent_extractor is not None:
        return _entity_extractor, _intent_extractor

    from utils import MultiLabelExtractor

    if os.path.exists(_ENTITY_MODEL_PATH) and os.path.exists(_INTENT_MODEL_PATH):
        logger.info("Loading ML extractors from disk...")
        _entity_extractor = MultiLabelExtractor.load(_ENTITY_MODEL_PATH)
        _intent_extractor = MultiLabelExtractor.load(_INTENT_MODEL_PATH)
    else:
        logger.warning("Model files not found — auto-training from dataset...")
        with open("dataset/entities.json") as f:
            entities = json.load(f)
        with open("dataset/intents.json") as f:
            intents = json.load(f)
        os.makedirs("models", exist_ok=True)
        _entity_extractor = MultiLabelExtractor()
        _entity_extractor.fit(entities, is_intent=False)
        _entity_extractor.save(_ENTITY_MODEL_PATH)
        _intent_extractor = MultiLabelExtractor()
        _intent_extractor.fit(intents, is_intent=True)
        _intent_extractor.save(_INTENT_MODEL_PATH)

    return _entity_extractor, _intent_extractor


# Ordered list: check longer/more-specific keywords first to avoid substring conflicts
_CABIN_KEYWORD_MAP: list[tuple[str, str]] = [
    ("eco standard",   "eco"),
    ("eco basic",      "promo"),
    ("sky boss",       "skyboss"),
    ("skyboss",        "skyboss"),
    ("hang thuong gia","skyboss"),
    ("hạng thương gia","skyboss"),
    ("ve skyboss",     "skyboss"),
    ("vé skyboss",     "skyboss"),
    ("hang deluxe",    "deluxe"),
    ("hạng deluxe",    "deluxe"),
    ("ve deluxe",      "deluxe"),
    ("vé deluxe",      "deluxe"),
    ("deluxe",         "deluxe"),
    ("hang eco",       "eco"),
    ("hạng eco",       "eco"),
    ("ve eco",         "eco"),
    ("vé eco",         "eco"),
    ("pho thong",      "eco"),
    ("phổ thông",      "eco"),
    ("hang promo",     "promo"),
    ("hạng promo",     "promo"),
    ("ve promo",       "promo"),
    ("vé promo",       "promo"),
    ("gia re",         "promo"),
    ("giá rẻ",         "promo"),
    ("eco",            "eco"),
    ("promo",          "promo"),
]


def _extract_cabin_class_keyword(text: str) -> str | None:
    """Keyword fallback khi ML extractor không nhận ra hạng vé (confidence dưới ngưỡng)."""
    lower = text.lower()
    for keyword, canonical in _CABIN_KEYWORD_MAP:
        if keyword in lower:
            return canonical
    return None


# ------------------------------------------------------------------
# Node: query type classification (command vs question)
# ------------------------------------------------------------------

async def query_type_node(state: GraphState) -> GraphState:
    logger.info("---QUERY TYPE NODE---")

    # Nếu đang chờ user nhập hạng vé (missing_slots từ lượt trước) → bỏ qua phân loại,
    # coi luôn là "question" để tiếp tục luồng faq_canonical_check
    if "cabin_class" in (state.get("missing_slots") or []):
        logger.info("Query type: question (slot continuation — waiting for cabin_class)")
        return {"query_type": "question"}

    result = predict_type_of_query(state["question"])
    query_type = "command" if result.get("command", 0) > result.get("question", 0) else "question"
    logger.info("Query type: %s | scores: %s", query_type, result)
    return {"query_type": query_type}


def _route_after_query_type(state: GraphState) -> str:
    return "command" if state.get("query_type") == "command" else "question"


async def command_not_supported_node(state: GraphState) -> GraphState:
    logger.info("---COMMAND NOT SUPPORTED NODE---")
    return {
        "final_answer": (
            "Hiện tại mình chưa hỗ trợ các yêu cầu đặt vé, đổi vé, hủy vé hay "
            "các thao tác đặt chỗ trực tiếp. Bạn vui lòng truy cập website hoặc "
            "ứng dụng VietJet Air để thực hiện nhé!"
        )
    }


# ------------------------------------------------------------------
# Node: check question against faq_data canonicals
# ------------------------------------------------------------------

async def faq_canonical_check_node(state: GraphState) -> GraphState:
    logger.info("---FAQ CANONICAL CHECK NODE---")
    question = state["question"]
    prev_canonical = state.get("faq_canonical_matched", "")
    prev_original_q = state.get("faq_original_question", "")

    # Extract entities trực tiếp trong node này (không cần entity_extraction_node)
    entity_ext, _ = _get_extractors()
    entities = entity_ext.predict_single(question, threshold=0.3)

    # Tìm canonical trong entities khớp với faq_data
    matched_canonical = ""
    for e in entities:
        if e.get("canonical") in _FAQ_CANONICALS:
            matched_canonical = e["canonical"]
            break

    # Nếu không tìm thấy canonical mới → tiếp tục dùng canonical cũ (multi-turn)
    if not matched_canonical:
        matched_canonical = prev_canonical

    # Lưu câu hỏi gốc khi bắt đầu topic mới
    if matched_canonical and matched_canonical != prev_canonical:
        faq_original_question = question
    else:
        faq_original_question = prev_original_q or question

    # Fill slots từ entities (đặc biệt cabin_class)
    filled: dict = dict(state.get("filled_slots") or {})
    for e in entities:
        slot_name = ENTITY_TYPE_TO_SLOT.get(e.get("type", ""))
        if slot_name:
            filled[slot_name] = e["canonical"]
    for slot, default_val in SLOT_DEFAULTS.items():
        if slot not in filled:
            filled[slot] = default_val

    # Fallback: ML extractor đôi khi cho cabin_class dưới ngưỡng 0.3 khi câu dài.
    # Dùng keyword match để đảm bảo không bỏ sót.
    if "cabin_class" not in filled:
        filled["cabin_class"] = _extract_cabin_class_keyword(question)

    logger.info("FAQ canonical: %s | cabin_class: %s", matched_canonical, filled.get("cabin_class"))

    return {
        "faq_canonical_matched": matched_canonical,
        "faq_original_question": faq_original_question,
        "filled_slots": filled,
        "entities": entities,
    }


def _route_after_faq_check(state: GraphState) -> str:
    if not state.get("faq_canonical_matched"):
        return "no_match"
    if not state.get("filled_slots", {}).get("cabin_class"):
        return "ask_cabin"
    return "proceed"


async def faq_not_found_node(state: GraphState) -> GraphState:
    logger.info("---FAQ NOT FOUND NODE---")
    return {
        "final_answer": (
            "Mình chưa ghi nhận được thông tin về vấn đề này trong hệ thống. "
            "Bạn vui lòng liên hệ trực tiếp VietJet để được hỗ trợ nhé!"
        )
    }


async def ask_cabin_class_node(state: GraphState) -> GraphState:
    logger.info("---ASK CABIN CLASS NODE---")
    q = "Bạn đang sử dụng hạng vé nào? (Promo / Eco / Deluxe / SkyBoss)"
    return {
        "final_answer": q,
        "missing_slots": ["cabin_class"],
        "slot_question": q,
    }


# ------------------------------------------------------------------
# Node: entity & intent extraction
# ------------------------------------------------------------------

async def entity_extraction_node(state: GraphState) -> GraphState:
    logger.info("---ENTITY EXTRACTION NODE---")
    from utils import detect_intent_mode

    question = state["question"]
    entity_ext, intent_ext = _get_extractors()

    matched_entities = entity_ext.predict_single(question, threshold=0.3)
    matched_intents  = intent_ext.predict_single(question, threshold=0.3)

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
    # Dùng câu hỏi gốc của topic (cho multi-turn: Turn 2 chỉ có "SkyBoss" nhưng topic là "xach_tay")
    question = state.get("faq_original_question") or state["question"]

    # Seed with entity canonicals, intent verb canonicals, resolved concepts, and faq canonical
    seed_ids: set[str] = set()
    faq_canonical = state.get("faq_canonical_matched", "")
    if faq_canonical:
        seed_ids.add(faq_canonical)
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
    # Dùng câu hỏi gốc cho BM25 (tránh trường hợp Turn 2 chỉ có "SkyBoss")
    question      = state.get("faq_original_question") or state["question"]
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
    hits = vector_store.similarity_search(
                question, k=VECTOR_K,
                filter=meta_filter if meta_filter else None,
            )
    all_pg_docs.extend(hits)

    # Deduplicate by content while preserving retrieval order
    seen_content: set[str] = set()
    pg_docs: list[Document] = []
    for doc in all_pg_docs:
        if doc.page_content not in seen_content:
            seen_content.add(doc.page_content)
            pg_docs.append(doc)

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
    print(best)
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
    # Dùng câu hỏi gốc của topic (đảm bảo đúng khi user chỉ trả lời "SkyBoss")
    question   = state.get("faq_original_question") or state["question"]
    docs       = state.get("documents", [])
    max_risk   = state.get("max_risk_level", 1)
    entities   = state.get("entities", [])
    intents    = state.get("intents", [])
    confidence = state.get("confidence", 0.0)
    suggested  = state.get("suggested_questions", [])
    print(confidence)
    # Low confidence or no retrieval → skip LLM entirely
    if not docs:
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
#   → query_type ──┬── (command)  → command_not_supported ──────────┐
#                  └── (question) → faq_canonical_check             │
#                                      ├── (no_match)  → faq_not_found ──┤
#                                      ├── (ask_cabin) → ask_cabin_class ─┤
#                                      └── (proceed)   → retrieve        │
#                                                           → assess_risk │
#                                                             → generate ─┘
#                                                                 → END
# ------------------------------------------------------------------
builder = StateGraph(GraphState)
builder.add_node("query_type",            query_type_node)
builder.add_node("command_not_supported", command_not_supported_node)
builder.add_node("faq_canonical_check",   faq_canonical_check_node)
builder.add_node("faq_not_found",         faq_not_found_node)
builder.add_node("ask_cabin_class",       ask_cabin_class_node)
builder.add_node("retrieve",              retrieve_node)
builder.add_node("assess_risk",           assess_risk_node)
builder.add_node("generate",              generate_node)

builder.add_edge(START, "query_type")
builder.add_conditional_edges(
    "query_type",
    _route_after_query_type,
    {"command": "command_not_supported", "question": "faq_canonical_check"},
)
builder.add_conditional_edges(
    "faq_canonical_check",
    _route_after_faq_check,
    {"no_match": "faq_not_found", "ask_cabin": "ask_cabin_class", "proceed": "retrieve"},
)
builder.add_edge("retrieve",              "assess_risk")
builder.add_edge("assess_risk",           "generate")
builder.add_edge("generate",              END)
builder.add_edge("ask_cabin_class",       END)
builder.add_edge("faq_not_found",         END)
builder.add_edge("command_not_supported", END)
