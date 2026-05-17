"""Stage 5: ingest chunks.jsonl into pgvector + persist BM25 sidecar.

Chunks are stored in the `vietjet_collection` table via langchain_postgres.
BM25 is kept as a pickle sidecar — pgvector does not support sparse retrieval
natively, and rebuilding from chunks on every restart is fast (< 1s).
"""

from __future__ import annotations
import json
import pickle
import re
import unicodedata
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from rank_bm25 import BM25Okapi

from vietjet.config import (
    BM25_PATH,
    CHUNKS_PATH,
    COLLECTION_NAME,
    DB_CONNECTION_STRING,
    EMBED_MODEL,
    INDEX_DIR,
)

TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def tokenize_vn(text: str) -> list[str]:
    return TOKEN_RE.findall(unicodedata.normalize("NFC", text).lower())


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def chunks_to_docs(chunks: list[dict]) -> list[Document]:
    return [
        Document(
            page_content=c["text"],
            metadata={
                "id": c["id"],
                "source": c["source"],
                "doc_type": c["doc_type"],
                "section_path": c["section_path"],
                "has_table": c["has_table"],
            },
        )
        for c in chunks
    ]


def ingest_pgvector(docs: list[Document], rebuild: bool = True) -> PGVector:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    store = PGVector(
        embeddings=embeddings,
        connection=DB_CONNECTION_STRING,
        collection_name=COLLECTION_NAME,
        use_jsonb=True,
        pre_delete_collection=rebuild,
    )
    ids = [d.metadata["id"] for d in docs]
    print(f"Embedding + upserting {len(docs)} docs into '{COLLECTION_NAME}'...")
    store.add_documents(docs, ids=ids)
    print(f"[OK] pgvector collection '{COLLECTION_NAME}' populated")
    return store


def persist_bm25(chunks: list[dict], path: Path = BM25_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenized = [tokenize_vn(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with path.open("wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks, "tokenized": tokenized}, f)
    print(f"[OK] BM25 sidecar → {path}")


def main(rebuild: bool = True) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")
    persist_bm25(chunks)
    docs = chunks_to_docs(chunks)
    ingest_pgvector(docs, rebuild=rebuild)


if __name__ == "__main__":
    main()
