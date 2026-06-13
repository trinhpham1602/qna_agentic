from __future__ import annotations

import asyncio
from typing import Any

from vietjet.cache import (
    compute_context_hash,
    get_cache_store,
    get_final_answer,
    get_redis_semantic_cache,
    is_realtime_intent,
    normalize_query_with_cache,
    set_final_answer,
    stable_hash,
)
from vietjet.cache.layers import get_query_embedding
from vietjet.config import (
    SEMANTIC_CACHE_ENABLED,
    SEMANTIC_CACHE_SKIP_REALTIME,
    SEMANTIC_CACHE_TENANT,
    TTL_FINAL_ANSWER,
    TTL_SEMANTIC_ANSWER,
)


def _docs_to_context_seed(docs: list) -> list[tuple[str, Any]]:
    seed: list[tuple[str, Any]] = []
    for d in docs or []:
        md = getattr(d, "metadata", {}) or {}
        doc_id = md.get("id") or md.get("source") or ""
        version = md.get("last_crawled_ts") or md.get("doc_type") or "static"
        seed.append((doc_id, version))
    return seed


def _doc_to_source(d) -> dict:
    md = getattr(d, "metadata", {}) or {}
    return {
        "source": md.get("source"),
        "section_path": md.get("section_path"),
        "doc_type": md.get("doc_type"),
        "url": md.get("source") if md.get("doc_type") == "web_live" else None,
    }


async def normalize_node(state: dict) -> dict:
    raw = state.get("question") or ""
    cache = get_cache_store()
    norm = await normalize_query_with_cache(cache, raw)
    realtime = is_realtime_intent(raw)
    return {
        "normalized_query": norm["normalized_query"],
        "slots": norm["slots"],
        "intent_realtime": realtime,
        "query": raw,
        "cached_from": None,
    }


async def get_embedding_node(state: dict) -> dict:
    from vietjet.config import EMBEDDING_MODEL_VERSION
    from vietjet.retrieval.embedder import get_embedder

    normalized = state.get("normalized_query") or state.get("question") or ""
    if not normalized:
        return {"query_embedding": None}

    cache = get_cache_store()

    embedder = get_embedder()

    def _embed_sync(text: str) -> list[float]:
        return list(embedder.embed_query(text))

    async def _embed_async(text: str) -> list[float]:
        return await asyncio.to_thread(_embed_sync, text)

    vector = await get_query_embedding(
        cache, normalized, _embed_async, model_name=EMBEDDING_MODEL_VERSION
    )
    return {"query_embedding": vector}


async def check_semantic_cache_node(state: dict) -> dict:
    if not SEMANTIC_CACHE_ENABLED:
        return {"cached_from": None}
    if SEMANTIC_CACHE_SKIP_REALTIME and state.get("intent_realtime"):
        return {"cached_from": None}

    embedding = state.get("query_embedding")
    if not embedding:
        return {"cached_from": None}

    sc = get_redis_semantic_cache()
    hit = await sc.lookup(embedding)
    if hit is None:
        return {"cached_from": None}

    print(
        f"[semantic-cache] HIT sim={hit.similarity:.3f} cached_q={hit.question[:60]!r}"
    )
    return {
        "cached_from": "semantic",
        "answer": hit.answer,
        "citations": [
            s.get("url") or s.get("source") or ""
            for s in hit.sources
            if s.get("url") or s.get("source")
        ],
        "context_hash": hit.context_hash,
    }


async def check_final_cache_node(state: dict) -> dict:
    if state.get("intent_realtime"):
        return {"cached_from": state.get("cached_from")}

    cache = get_cache_store()
    normalized = state.get("normalized_query") or ""
    if not normalized:
        return {"cached_from": state.get("cached_from")}

    hit = await get_final_answer(
        cache, SEMANTIC_CACHE_TENANT, "public", normalized
    )
    if hit is None:
        return {"cached_from": state.get("cached_from")}

    print(f"[final-cache] HIT for normalized={normalized[:60]!r}")
    return {
        "cached_from": "final",
        "answer": hit.get("answer", ""),
        "citations": hit.get("citations") or [],
        "context_hash": hit.get("context_hash"),
    }


def after_cache_check(state: dict) -> str:
    if state.get("cached_from"):
        return "return_cached"
    return "route"


async def return_cached_node(state: dict) -> dict:
    return {
        "answer": state.get("answer") or "",
        "citations": state.get("citations") or [],
    }


async def store_cache_node(state: dict) -> dict:
    if state.get("cached_from"):
        return {}
    if state.get("intent_realtime"):
        return {}

    answer = state.get("answer") or ""
    if not answer.strip():
        return {}

    normalized = state.get("normalized_query") or ""
    if not normalized:
        return {}

    db_docs = state.get("docs") or []
    web_docs = state.get("web_docs") or []
    all_docs = list(db_docs) + list(web_docs)
    if not all_docs:
        return {}

    context_hash = compute_context_hash(_docs_to_context_seed(all_docs))
    sources = [_doc_to_source(d) for d in all_docs]

    cache = get_cache_store()
    await set_final_answer(
        cache,
        SEMANTIC_CACHE_TENANT,
        "public",
        normalized,
        {
            "answer": answer,
            "citations": state.get("citations") or [],
            "context_hash": context_hash,
            "sources": sources,
        },
        ttl=TTL_FINAL_ANSWER,
    )

    embedding = state.get("query_embedding")
    if embedding:
        sc = get_redis_semantic_cache()
        await sc.store(
            question=state.get("question") or "",
            normalized_query=normalized,
            answer=answer,
            sources=sources,
            context_hash=context_hash,
            embedding=embedding,
            ttl_seconds=TTL_SEMANTIC_ANSWER,
        )

    return {"context_hash": context_hash}
