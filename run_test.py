import asyncio
from typing import List
from langchain_core.documents import Document
from typing import TypedDict
import asyncio
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import AsyncChromiumLoader


from config import (
    LLM_MODEL
)

urls = [
    "https://www.vietjetair.com/vi/pages/de-co-chuyen-bay-tot-dep-1578323501979/dieu-kien-ve-1641466500765",
    "https://www.vietjetair.com/vi/pages/de-co-chuyen-bay-tot-dep-1578323501979/phi-va-le-phi-1578483039924",
    "https://www.vietjetair.com/vi/pages/de-co-chuyen-bay-tot-dep-1578323501979/quy-dinh-hanh-ly-1578483259803",
    "https://www.vietjetair.com/vi/pages/de-co-chuyen-bay-tot-dep-1578323501979/giay-to-tuy-than-1578483122906",
]

# docs = [AsyncChromiumLoader(url).load() for url in urls]
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# docs_list = [item for sublist in docs for item in sublist]
# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=100, chunk_overlap=50
# )
# doc_splits = text_splitter.split_documents(docs_list)

from langchain_huggingface import HuggingFaceEmbeddings
from config import (
    DB_CONNECTION_STRING,
    EMBEDDING_MODEL,
    LLM_MODEL,
    COLLECTION_NAME,
    VECTOR_K,
)
from langchain_postgres import PGVector

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="partner_link",
    connection=DB_CONNECTION_STRING,
    use_jsonb=True,
)

class ConversationState(TypedDict):
    question: str
    docs: List[Document]
    final_answer: str

from langgraph.graph import StateGraph, START, END
builder = StateGraph(ConversationState)
async def retrieve_node(state: ConversationState) -> ConversationState:
    question = state.get("question", "")
    hits = vector_store.similarity_search(
                question, k=VECTOR_K,
                filter=None,
            )
    return {"docs": hits}
async def generate_node(state: ConversationState) -> ConversationState:
    question   = state["question"]
    docs       = state.get("docs", [])
    llm = ChatOllama(model=LLM_MODEL, temperature=0, num_predict=320)
    context  = "\n---\n".join(doc.page_content for doc in docs)
    sys_prompt = (
        "Bạn là trợ lý CSKH thân thiện, nhiệt tình. Trả lời HOÀN TOÀN bằng tiếng Việt.\n"
        "Giọng điệu ấm áp, gần gũi — dùng 'bạn', xưng 'mình', thêm 'nhé'/'ạ' khi phù hợp.\n"
        "CHỈ dùng thông tin từ tài liệu bên dưới. Nếu không có, nói: 'Thông tin này mình chưa tìm thấy, bạn liên hệ hotline để được hỗ trợ thêm nhé.'\n"
        "Dùng thẻ HTML <b>text</b> để in đậm từ quan trọng. KHÔNG dùng markdown **.\n"
        f"Tài liệu tham khảo:\n{context}\n"
        "Trả lời ngắn gọn, dễ hiểu."
    )
    try:
        response = await llm.ainvoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=question),
        ])
        answer = response.content
        print(answer)
    except Exception as e:
        answer = "Xin lỗi, hệ thống đang quá tải. Vui lòng thử lại sau."

    return {
        "final_answer": answer,
    }

builder.add_node("retrieve", retrieve_node)
builder.add_node("generate_node", generate_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate_node")
builder.add_edge("generate_node", END)

# ===== DEFINE LABEL SPACE =====
async def off_topic_node():
    llm = ChatOllama(model=LLM_MODEL, temperature=0.3, num_predict=320)
    sys_prompt = (
        "Bạn là một trợ lý trích xuất nhiều đối tượng và nhiều mục đích. Trả lời bằng định dạng json"
    )
    try:
        response = await llm.ainvoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content="toi muốn đổi vé thì làm như thế nào"),
        ])
        answer = response.content
        print(answer)
    except Exception as e:
        answer = "Xin lỗi, mình chưa thể trả lời câu hỏi này lúc này."
    return {"final_answer": answer}


async def off_topic_node_v2():
    llm = ChatOllama(model=LLM_MODEL, temperature=0.3, num_predict=320)
    sys_prompt = (
        "Bạn là trợ lý xem xét những câu hỏi của user và suy luận được những câu hỏi có khả năng liên quan cao đến câu hỏi của user. Cho tôi 3 câu hỏi có khả năng cao nhất."
    )
    try:
        response = await llm.ainvoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content="Tôi muốn đổi vé thì làm như thế nào"),
        ])
        answer = response.content
        print(answer)
    except Exception as e:
        answer = "Xin lỗi, mình chưa th"

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
if __name__ == "__main__":
    asyncio.run(off_topic_node_v2())
