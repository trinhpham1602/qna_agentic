import asyncio
from typing import Dict, TypedDict, List, Annotated
import json
import logging

from psycopg_pool import ConnectionPool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

# Import the Ollama client user already had in main.py or directly
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 1. SETUP VECTOR STORE & SQL PSEUDO-CODE
# ---------------------------------------------------------
# Assume this is the connection string to PostgreSQL with pgvector extension enabled.
DB_CONNECTION_STRING = "postgresql://admin:123456@localhost:5432/rag_db"

def init_postgres_db():
    """
    Pseudo-code function to show how the table and extensions are created.
    You need to run this on your real Postgres instance.
    """
    create_table_sql = """
    -- Enable pgvector extension
    CREATE EXTENSION IF NOT EXISTS vector;

    -- Create table for storing FAQ documents
    CREATE TABLE IF NOT EXISTS faq_documents (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        content TEXT,
        metadata JSONB,
        embedding vector(384) -- 384 is the dimension for all-MiniLM-L6-v2
    );

    -- Create an index for faster similarity search
    CREATE INDEX IF NOT EXISTS faq_documents_embedding_idx ON faq_documents 
    USING hnsw (embedding vector_cosine_ops);
    """
    logger.info("SQL TO RUN IN POSTGRESQL:")
    logger.info(create_table_sql)
    
    # In a real setup, we execute this using psycopg pool:
    with ConnectionPool(DB_CONNECTION_STRING) as pool:
        with pool.connection() as conn:
            conn.execute(create_table_sql)

# Initialize Embeddings model
# Using keepitreal/vietnamese-sbert as it's optimized for Vietnamese
embeddings = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")

# Mocking the VectorStore interactions since we don't have a real DB running here.
# In reality, you could use Langchain's PGVector:
from langchain_postgres import PGVector
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="faq_collection_vn", # Changed name to avoid dimension conflict with the previous 384 dim model
    connection=DB_CONNECTION_STRING,
    use_jsonb=True,
)

# Global BM25 retriever
bm25_retriever = None


def ingest_faq_data():
    global bm25_retriever
    from data import faq_data
    from langchain_community.retrievers import BM25Retriever
    docs = []
    for item in faq_data:
        # Create a document for each FAQ
        content = f"Tình huống: {item.get('situation')}\n" \
                  f"Quy định Eco: {item.get('eco_standard')}\n" \
                  f"Quy định Promo: {item.get('promo_eco_basic')}\n" \
                  f"Quy định Deluxe: {item.get('deluxe')}\n" \
                  f"Quy định SkyBoss: {item.get('skyboss')}\n" \
                  f"Ghi chú: {item.get('important_notes')}\n" \
                  f"Trả lời mẫu: {item.get('short_answer')}"
        doc = Document(
            page_content=content, 
            metadata={
                "category": item.get('category'),
                "risk": item.get('risk'),
                "escalate_when": item.get('escalate_when')
            }
        )
        docs.append(doc)
    
    # Initialize BM25 with our documents
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3
    
    vector_store.add_documents(docs)

# ---------------------------------------------------------
# 2. LANGGRAPH RAG WORKFLOW DEFINITION
# ---------------------------------------------------------

class GraphState(TypedDict):
    """State of the RAG workflow"""
    question: str
    documents: List[Document]
    max_risk_level: int
    final_answer: str
    conversation_summary: str

async def retrieve_node(state: GraphState) -> GraphState:
    logger.info("---RETRIEVAL NODE---")
    question = state["question"]
    
    # 1. Retrieve from PGVector (Semantic)
    try:
        pg_docs = vector_store.similarity_search(question, k=3)
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        pg_docs = []
        
    # 2. Retrieve from BM25 (Keyword)
    bm25_docs = bm25_retriever.invoke(question) if bm25_retriever else []
    
    # 3. Reciprocal Rank Fusion (RRF)
    fused_scores = {}
    doc_map = {}
    
    for doc_list in [pg_docs, bm25_docs]:
        for rank, doc in enumerate(doc_list):
            doc_str = doc.page_content
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
                doc_map[doc_str] = doc
            fused_scores[doc_str] += 1 / (rank + 60)
            
    # Sort docs by fused score
    reranked_docs = [
        doc_map[content] 
        for content, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    
    # Return top 2 unified docs conceptually, but expand them:
    best_docs = reranked_docs[:2]
    if not best_docs:
        return {"documents": []}
        
    top_category = best_docs[0].metadata.get("category")
    logger.info(f"Detected Category: {top_category}")
    
    # Expand context to include ALL situations in this category
    from data import faq_data
    expanded_docs = []
    for item in faq_data:
        if item.get("category") == top_category:
            content = f"Tình huống: {item.get('situation')}\n" \
                      f"Quy định Eco: {item.get('eco_standard')}\n" \
                      f"Quy định Promo: {item.get('promo_eco_basic')}\n" \
                      f"Quy định Deluxe: {item.get('deluxe')}\n" \
                      f"Quy định SkyBoss: {item.get('skyboss')}\n" \
                      f"Ghi chú: {item.get('important_notes')}\n" \
                      f"Trả lời mẫu: {item.get('short_answer')}"
            
            doc = Document(
                page_content=content,
                metadata={
                    "category": item.get('category'),
                    "risk": item.get('risk'),
                    "escalate_when": item.get('escalate_when')
                }
            )
            expanded_docs.append(doc)
            
    logger.info(f"Expanded to {len(expanded_docs)} documents covering all situations in category '{top_category}'")
    
    return {"documents": expanded_docs}

async def assess_risk_node(state: GraphState) -> GraphState:
    logger.info("---ASSESS RISK NODE---")
    docs = state.get("documents", [])
    
    max_risk = 0
    for doc in docs:
        risk_level = doc.metadata.get("risk", 0)
        if risk_level > max_risk:
            max_risk = risk_level
            
    logger.info(f"Fuzzy Risk Assessed -> Max Risk Level: {max_risk}")
    return {"max_risk_level": max_risk}

async def escalate_api_action() -> str:
    """Standalone API call logic for the /escalate endpoint"""
    logger.info("---ESCALATION ACTION (ASYNC)---")
    # Simulate an external API call to Customer Service System
    logger.info("Calling External Escalate API... (waiting 2s)")
    await asyncio.sleep(2) 
    logger.info("Escalate API call Success!")
    
    answer = "Hệ thống đã kết nối thành công. Một nhân viên CSKH sẽ tiếp nhận và phản hồi bạn trong giây lát."
    return answer

async def generate_node(state: GraphState) -> GraphState:
    logger.info("---GENERATE NODE---")
    question = state["question"]
    docs = state["documents"]
    max_risk = state.get("max_risk_level", 0)
    convo_summary = state.get("conversation_summary", "")
    
    # Construct context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    llm = ChatOllama(model="llama3.1", temperature=0)
    
    sys_prompt = f"Bạn là trợ lý CSKH. Dưới đây là các bộ quy định (gồm các tình huống chung):\n{context}\n\n"
    if convo_summary:
        sys_prompt += f"[LỊCH SỬ TRÒ CHUYỆN ĐẾN HIỆN TẠI: {convo_summary}]\n\n"
        
    sys_prompt += f"Mức độ rủi ro hệ thống đánh giá cho query này là: {max_risk}/3.\n" \
                 f"Hãy trả lời câu hỏi trực tiếp, giúp khách hàng nắm rõ quy định. "
    
    if max_risk >= 3:
        sys_prompt += "ĐẶC BIỆT LƯU Ý BẮT BUỘC: Vì đây là tác vụ rủi ro cao hoặc liên quan tài chính, NGƯƠI PHẢI nhắc khách hàng bấm nút Tùy chọn 'Xác nhận Chuyển nhân viên CSKH' ở phía dưới để được hỗ trợ chuyên sâu."
    elif max_risk == 2:
        sys_prompt += "Lưu ý thêm: Nếu khách hàng gặp khó khăn, hãy gợi ý nhẹ nhàng rằng họ có thể bấm nút Tùy chọn chuyển gặp nhân viên phía dưới."
    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=question)
    ]
    
    logger.info("Calling LLM...")
    # Mock response if LLM runs slow or fails, but we try real ollama call
    try:
        response = await llm.ainvoke(messages)
        answer = response.content
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        answer = "Xin lỗi, hệ thống tạo câu trả lời đang quá tải. (Fallback Message: " + docs[0].metadata.get("short_answer", "") + ")"
    
    return {"final_answer": answer}


async def summarize_memory_node(state: GraphState) -> GraphState:
    logger.info("---SUMMARIZE MEMORY NODE---")
    question = state["question"]
    final_answer = state["final_answer"]
    old_summary = state.get("conversation_summary", "")
    
    llm = ChatOllama(model="llama3.1", temperature=0)
    sys_prompt = f"Bạn là một con AI đúc kết hội thoại. Hãy viết một đoạn tóm tắt ngắn (dưới 50 từ) về bối cảnh đang thảo luận dựa trên tóm tắt cũ và tin nhắn mới nhất này.\nTóm tắt cũ: {old_summary}\nNgười dùng hỏi: {question}\nBot đáp: {final_answer}\nViết tóm tắt mới:"
    
    try:
        response = await llm.ainvoke([SystemMessage(content=sys_prompt)])
        new_summary = response.content
    except Exception as e:
        logger.error(f"Summarize Error: {e}")
        new_summary = old_summary
        
    return {"conversation_summary": new_summary}

# Build Graph
builder = StateGraph(GraphState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("assess_risk", assess_risk_node)
builder.add_node("generate", generate_node)
builder.add_node("summarize_memory", summarize_memory_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "assess_risk")
builder.add_edge("assess_risk", "generate")
builder.add_edge("generate", "summarize_memory")
builder.add_edge("summarize_memory", END)

# Setup is done natively using psycopg.connect as a fallback for table creation
# We only export the builder here.
# FastAPI will compile the graph with the Async checkpointer inside its lifespan.

# Automatically init db and ingest data when imported
init_postgres_db()
ingest_faq_data()
