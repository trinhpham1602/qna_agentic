import os
import uuid
import tempfile
import logging
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List as _List
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from config import DB_CONNECTION_STRING
from document_ingester import DocumentIngester
from rag_workflow import (
    builder,
    escalate_api_action,
    init_postgres_db,
    ingest_default_faq,
    add_documents,
    get_stats,
)

logger = logging.getLogger(__name__)

compiled_rag_graph = None
ingester = DocumentIngester()

SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".pdf", ".docx", ".doc", ".txt", ".csv", ".json"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global compiled_rag_graph

    # One-time setup: DB tables + default FAQ data
    init_postgres_db()
    ingest_default_faq()

    async with AsyncConnectionPool(DB_CONNECTION_STRING) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        compiled_rag_graph = builder.compile(checkpointer=checkpointer)

        try:
            img = compiled_rag_graph.get_graph().draw_mermaid_png()
            with open("graph.png", "wb") as f:
                f.write(img)
        except Exception:
            pass

        yield


# ------------------------------------------------------------------
# Request / Response schemas
# ------------------------------------------------------------------

class ThreadResponse(BaseModel):
    thread_id: str


class ChatRequest(BaseModel):
    thread_id: str
    query: str


class ChatResponse(BaseModel):
    answer: str
    suggest_escalate: bool = False
    confidence: float = 1.0
    suggested_questions: _List[str] = []
    matched_entities: _List[str] = []
    matched_intents: _List[str] = []
    slot_question: str = ""
    missing_slots: _List[str] = []
    reconstructed_query: str = ""
    is_off_topic: bool = False


class EscalateResponse(BaseModel):
    status: str
    message: str


class IngestResponse(BaseModel):
    status: str
    documents_added: int
    source: str


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

app = FastAPI(
    title="Customer Service RAG API",
    description="LangGraph RAG with flexible document ingestion",
    version="2.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------
# Chat endpoints
# ------------------------------------------------------------------

@app.post("/thread", response_model=ThreadResponse)
async def create_thread():
    """Create a new conversation thread."""
    return ThreadResponse(thread_id=str(uuid.uuid4()))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a user query through the RAG + LangGraph pipeline.
    Returns the answer and whether to suggest escalation.
    """
    config = {"configurable": {"thread_id": request.thread_id}}
    final_state = await compiled_rag_graph.ainvoke(
        {"question": request.query}, config=config
    )
    return ChatResponse(
        answer=final_state["final_answer"],
        suggest_escalate=final_state.get("max_risk_level", 0) >= 8,
        confidence=round(final_state.get("confidence", 1.0), 3),
        suggested_questions=final_state.get("suggested_questions", []),
        matched_entities=[
            e.get("label") or e["canonical"]
            for e in final_state.get("entities", [])
        ],
        matched_intents=[
            i.get("label") or i["canonical"]
            for i in final_state.get("intents", [])
        ],
        slot_question=final_state.get("slot_question", ""),
        missing_slots=final_state.get("missing_slots", []),
        reconstructed_query=final_state.get("reconstructed_query", ""),
        is_off_topic=final_state.get("is_off_topic", False),
    )


@app.post("/escalate", response_model=EscalateResponse)
async def escalate():
    """Trigger a human escalation after user confirms."""
    message = await escalate_api_action()
    return EscalateResponse(status="success", message=message)


# ------------------------------------------------------------------
# Document management endpoints
# ------------------------------------------------------------------

@app.post("/documents/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    category: Optional[str] = Form(None),
    risk: Optional[int] = Form(None),
):
    """
    Upload a file and ingest it into the RAG system.

    Supported formats: xlsx, xls, pdf, docx, doc, txt, csv, json.
    Optional form fields:
      - category: override/set the document category metadata
      - risk: set risk level (1-3) for all ingested chunks
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        extra: dict = {}
        if category:
            extra["category"] = category
        if risk is not None:
            extra["risk"] = risk

        docs = ingester.load(tmp_path, **extra)
        add_documents(docs)
        return IngestResponse(status="ok", documents_added=len(docs), source=file.filename)
    finally:
        os.unlink(tmp_path)


@app.post("/documents/url", response_model=IngestResponse)
async def ingest_url(
    url: str,
    category: Optional[str] = None,
    risk: Optional[int] = None,
):
    """
    Fetch a URL and ingest its text content into the RAG system.

    Query params:
      - url: target URL (required)
      - category: metadata category label
      - risk: risk level 1-3
    """
    extra: dict = {}
    if category:
        extra["category"] = category
    if risk is not None:
        extra["risk"] = risk

    docs = ingester.load(url, **extra)
    add_documents(docs)
    return IngestResponse(status="ok", documents_added=len(docs), source=url)


@app.get("/documents/stats")
async def document_stats():
    """Return total document count and breakdown by category."""
    return get_stats()


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
