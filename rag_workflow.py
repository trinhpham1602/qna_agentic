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
    question:            str
    entities:            List[dict]   # extracted airline / cabin / route / service / passenger
    intents:             List[dict]   # extracted intents with Fibonacci risk
    graph_risk:          int          # max Fibonacci risk from matched intents
    documents:           List[Document]
    max_risk_level:      int
    final_answer:        str
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

    graph_risk = max((i.get("risk", 1) for i in matched_intents), default=1)

    logger.info("Entities : %s", [e["canonical"] for e in matched_entities])
    logger.info("Intents  : %s", [(i["canonical"], i.get("risk")) for i in matched_intents])
    logger.info("Graph risk (Fibonacci): %d", graph_risk)

    return {
        "entities":   matched_entities,
        "intents":    matched_intents,
        "graph_risk": graph_risk,
    }


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
    question = state["question"]
    docs     = state.get("documents", [])
    max_risk = state.get("max_risk_level", 1)
    summary  = state.get("conversation_summary", "")
    entities = state.get("entities", [])
    intents  = state.get("intents", [])

    if not docs:
        return {
            "final_answer": (
                "Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn. "
                "Vui lòng liên hệ hotline VietJet để được hỗ trợ trực tiếp."
            )
        }

    context = "\n\n".join(doc.page_content for doc in docs)
    llm     = ChatOllama(model=LLM_MODEL, temperature=0)

    entity_ctx = ""
    if entities or intents:
        ent_labels  = [e.get("label") or e["canonical"] for e in entities]
        intent_labels = [i.get("label") or i["canonical"] for i in intents]
        entity_ctx  = f"[Thực thể: {', '.join(ent_labels)}] [Mục đích: {', '.join(intent_labels)}]\n"

    sys_prompt = (
        f"Bạn là trợ lý CSKH. {entity_ctx}"
        f"Dưới đây là các quy định liên quan:\n{context}\n\n"
    )
    if summary:
        sys_prompt += f"[LỊCH SỬ TRÒ CHUYỆN: {summary}]\n\n"
    sys_prompt += f"Mức độ rủi ro: {max_risk} (Fibonacci scale). Trả lời trực tiếp, rõ ràng. "

    if max_risk >= RISK_ESCALATE_HARD:
        sys_prompt += (
            "BẮT BUỘC: Đây là tác vụ rủi ro cao/tài chính. "
            "PHẢI nhắc khách hàng bấm 'Xác nhận Chuyển nhân viên CSKH' để được hỗ trợ chuyên sâu."
        )
    elif max_risk >= RISK_ESCALATE_SOFT:
        sys_prompt += "Gợi ý nhẹ nhàng rằng họ có thể chuyển gặp nhân viên nếu cần."

    try:
        response = await llm.ainvoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=question),
        ])
        answer = response.content
    except Exception as e:
        logger.error(f"LLM error: {e}")
        answer = "Xin lỗi, hệ thống đang quá tải. Vui lòng thử lại sau."

    return {"final_answer": answer}


# ------------------------------------------------------------------
# Node: conversation memory summary
# ------------------------------------------------------------------

async def summarize_memory_node(state: GraphState) -> GraphState:
    logger.info("---SUMMARIZE MEMORY NODE---")
    old_summary = state.get("conversation_summary", "")
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    prompt = (
        f"Viết tóm tắt ngắn (dưới 50 từ) bối cảnh đang thảo luận.\n"
        f"Tóm tắt cũ: {old_summary}\n"
        f"Người dùng hỏi: {state['question']}\n"
        f"Bot đáp: {state['final_answer']}\n"
        f"Tóm tắt mới:"
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
# ------------------------------------------------------------------
builder = StateGraph(GraphState)
builder.add_node("entity_extraction", entity_extraction_node)
builder.add_node("retrieve",          retrieve_node)
builder.add_node("assess_risk",       assess_risk_node)
builder.add_node("generate",          generate_node)
builder.add_node("summarize_memory",  summarize_memory_node)

builder.add_edge(START,              "entity_extraction")
builder.add_edge("entity_extraction", "retrieve")
builder.add_edge("retrieve",          "assess_risk")
builder.add_edge("assess_risk",       "generate")
builder.add_edge("generate",          "summarize_memory")
builder.add_edge("summarize_memory",  END)
