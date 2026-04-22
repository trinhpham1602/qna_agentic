import os

DB_CONNECTION_STRING = os.getenv("DATABASE_URL", "postgresql://admin:123456@localhost:5432/rag_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "keepitreal/vietnamese-sbert")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "faq_collection_vn")
VECTOR_K = int(os.getenv("VECTOR_K", "3"))
BM25_K = int(os.getenv("BM25_K", "3"))
