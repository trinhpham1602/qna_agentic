from __future__ import annotations

from typing import Any, Awaitable, Callable, Iterable, Optional

from vietjet.cache.store import CacheStore, stable_hash
from vietjet.config import (
    COLLECTION_VERSION,
    EMBEDDING_MODEL_VERSION,
    RETRIEVER_VERSION,
    TTL_CHUNK,
    TTL_DOC,
    TTL_EMBEDDING,
    TTL_FINAL_ANSWER,
    TTL_RETRIEVAL,
    TTL_WEB_PAGE_OFFICIAL,
    TTL_WEB_SEARCH_NORMAL,
)


async def get_query_embedding(
    cache: CacheStore,
    normalized_query: str,
    embed_fn: Callable[[str], Awaitable[list[float]] | list[float]],
    *,
    model_name: str = EMBEDDING_MODEL_VERSION,
    ttl: int = TTL_EMBEDDING,
) -> list[float]:
    h = stable_hash(normalized_query)
    key = cache.build_key("emb", model_name, h)

    cached = await cache.get_json(key)
    if cached and "vector" in cached:
        return cached["vector"]

    result = embed_fn(normalized_query)
    vector = await result if hasattr(result, "__await__") else result

    await cache.set_json(
        key,
        {"query": normalized_query, "vector": list(vector), "model": model_name},
        ttl_seconds=ttl,
    )
    return vector


async def retrieve_with_cache(
    cache: CacheStore,
    normalized_query: str,
    retrieve_fn: Callable[[str], Awaitable[list] | list],
    *,
    collection: str = COLLECTION_VERSION,
    retriever_version: str = RETRIEVER_VERSION,
    ttl: int = TTL_RETRIEVAL,
    extract_chunk_id: Callable[[Any], str] | None = None,
    extract_score: Callable[[Any], float | None] | None = None,
) -> dict:
    h = stable_hash(normalized_query)
    key = cache.build_key("retrieval", collection, retriever_version, h)

    cached = await cache.get_json(key)
    if cached:
        return cached

    result = retrieve_fn(normalized_query)
    docs = await result if hasattr(result, "__await__") else result

    if extract_chunk_id is None:
        def extract_chunk_id(d):
            md = getattr(d, "metadata", {}) or {}
            return md.get("chunk_id") or md.get("source") or ""
    if extract_score is None:
        def extract_score(d):
            md = getattr(d, "metadata", {}) or {}
            return md.get("score") or md.get("rerank_score")

    payload = {
        "normalized_query": normalized_query,
        "chunk_ids": [extract_chunk_id(d) for d in docs],
        "scores": [extract_score(d) for d in docs],
        "count": len(docs),
    }
    await cache.set_json(key, payload, ttl_seconds=ttl)
    return payload


async def get_doc_cached(
    cache: CacheStore, doc_id: str, version: str
) -> Optional[dict]:
    key = cache.build_key("doc", doc_id, version)
    return await cache.get_json(key)


async def set_doc_cached(
    cache: CacheStore, doc_id: str, version: str, payload: dict, *, ttl: int = TTL_DOC
) -> bool:
    key = cache.build_key("doc", doc_id, version)
    return await cache.set_json(key, payload, ttl_seconds=ttl)


async def get_chunk_cached(
    cache: CacheStore, chunk_id: str, version: str
) -> Optional[dict]:
    key = cache.build_key("chunk", chunk_id, version)
    return await cache.get_json(key)


async def set_chunk_cached(
    cache: CacheStore,
    chunk_id: str,
    version: str,
    payload: dict,
    *,
    ttl: int = TTL_CHUNK,
) -> bool:
    key = cache.build_key("chunk", chunk_id, version)
    return await cache.set_json(key, payload, ttl_seconds=ttl)


async def web_search_with_cache(
    cache: CacheStore,
    normalized_query: str,
    search_fn: Callable[[str], Awaitable[list] | list],
    *,
    provider: str = "tavily",
    ttl: int = TTL_WEB_SEARCH_NORMAL,
) -> list:
    h = stable_hash(normalized_query)
    key = cache.build_key("websearch", provider, "v1", h)

    cached = await cache.get_json(key)
    if cached and "results" in cached:
        return cached["results"]

    result = search_fn(normalized_query)
    results = await result if hasattr(result, "__await__") else result

    await cache.set_json(
        key,
        {"query": normalized_query, "provider": provider, "results": results},
        ttl_seconds=ttl,
    )
    return results


async def webpage_with_cache(
    cache: CacheStore,
    url: str,
    fetch_fn: Callable[[str], Awaitable[dict] | dict],
    *,
    ttl: int = TTL_WEB_PAGE_OFFICIAL,
) -> dict:
    h = stable_hash(url)
    key = cache.build_key("webpage", "v1", h)

    cached = await cache.get_json(key)
    if cached:
        return cached

    result = fetch_fn(url)
    page = await result if hasattr(result, "__await__") else result

    payload = {
        "url": url,
        "title": page.get("title"),
        "content": page.get("content"),
        "fetched_at": page.get("fetched_at"),
        "content_hash": stable_hash(page.get("content", "")),
    }
    await cache.set_json(key, payload, ttl_seconds=ttl)
    return payload


def build_final_answer_key(
    cache: CacheStore,
    tenant: str,
    user_scope: str,
    normalized_query: str,
) -> str:
    h = stable_hash(normalized_query)
    return cache.build_key("answer", tenant, user_scope, h)


async def get_final_answer(
    cache: CacheStore,
    tenant: str,
    user_scope: str,
    normalized_query: str,
    expected_context_hash: str | None = None,
) -> Optional[dict]:
    key = build_final_answer_key(cache, tenant, user_scope, normalized_query)
    cached = await cache.get_json(key)
    if not cached:
        return None
    if expected_context_hash is not None and cached.get("context_hash") != expected_context_hash:
        return None
    return cached


async def set_final_answer(
    cache: CacheStore,
    tenant: str,
    user_scope: str,
    normalized_query: str,
    payload: dict,
    *,
    ttl: int = TTL_FINAL_ANSWER,
) -> bool:
    key = build_final_answer_key(cache, tenant, user_scope, normalized_query)
    return await cache.set_json(key, payload, ttl_seconds=ttl)


def compute_context_hash(doc_ids_with_versions: Iterable[tuple[str, Any]]) -> str:
    payload = [f"{doc_id}:{ver}" for doc_id, ver in doc_ids_with_versions]
    return stable_hash(payload)
