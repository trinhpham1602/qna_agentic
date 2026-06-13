from __future__ import annotations

import json
import uuid
from typing import Optional

import numpy as np

from vietjet.cache.semantic import SemanticHit
from vietjet.cache.store import get_cache_store
from vietjet.config import (
    REDIS_KEY_PREFIX,
    SEMANTIC_CACHE_REDIS_THRESHOLD,
    SEMANTIC_CACHE_TENANT,
    TTL_SEMANTIC_ANSWER,
)


def _cosine(a, b) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class RedisSemanticCache:
    def __init__(
        self,
        *,
        tenant_id: str = SEMANTIC_CACHE_TENANT,
        threshold: float = SEMANTIC_CACHE_REDIS_THRESHOLD,
        prefix: str = REDIS_KEY_PREFIX,
    ) -> None:
        self.tenant_id = tenant_id
        self.threshold = threshold
        self.prefix = prefix

    @property
    def _redis(self):
        return get_cache_store().redis

    def _entry_key(self, user_scope: str, entry_id: str) -> str:
        return f"{self.prefix}:semcache:e:{self.tenant_id}:{user_scope}:{entry_id}"

    def _index_key(self, user_scope: str) -> str:
        return f"{self.prefix}:semcache:idx:{self.tenant_id}:{user_scope}"

    async def lookup(
        self,
        embedding: list[float],
        *,
        user_scope: str = "public",
        expected_context_hash: Optional[str] = None,
    ) -> Optional[SemanticHit]:
        if not embedding:
            return None
        index_key = self._index_key(user_scope)
        try:
            entry_ids = await self._redis.smembers(index_key)
        except Exception as exc:
            print(f"[semcache-redis] smembers failed: {exc}")
            return None
        if not entry_ids:
            return None

        ids = list(entry_ids)
        keys = [self._entry_key(user_scope, i) for i in ids]
        try:
            raws = await self._redis.mget(keys)
        except Exception as exc:
            print(f"[semcache-redis] mget failed: {exc}")
            return None

        best_entry: Optional[dict] = None
        best_sim = -1.0
        stale: list[str] = []
        for entry_id, raw in zip(ids, raws):
            if raw is None:
                stale.append(entry_id)
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                stale.append(entry_id)
                continue
            emb = entry.get("embedding")
            if not emb:
                continue
            sim = _cosine(embedding, emb)
            if sim < self.threshold:
                continue
            if (
                expected_context_hash is not None
                and entry.get("context_hash") != expected_context_hash
            ):
                continue
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if stale:
            try:
                await self._redis.srem(index_key, *stale)
            except Exception:
                pass

        if best_entry is None:
            return None

        print(
            f"[semcache-redis] HIT sim={best_sim:.3f} "
            f"cached_q={best_entry.get('question', '')[:60]!r}"
        )
        return SemanticHit(
            id=0,
            question=best_entry.get("question", ""),
            answer=best_entry.get("answer", ""),
            sources=best_entry.get("sources") or [],
            context_hash=best_entry.get("context_hash", ""),
            similarity=best_sim,
            normalized_query=best_entry.get("normalized_query", ""),
        )

    async def store(
        self,
        *,
        question: str,
        normalized_query: str,
        answer: str,
        sources: list,
        context_hash: str,
        embedding: list[float],
        user_scope: str = "public",
        ttl_seconds: int = TTL_SEMANTIC_ANSWER,
    ) -> Optional[str]:
        if not embedding:
            return None
        entry_id = uuid.uuid4().hex
        entry = {
            "question": question,
            "normalized_query": normalized_query,
            "answer": answer,
            "sources": sources,
            "context_hash": context_hash,
            "embedding": list(embedding),
        }
        entry_key = self._entry_key(user_scope, entry_id)
        index_key = self._index_key(user_scope)
        try:
            await self._redis.set(
                entry_key, json.dumps(entry, ensure_ascii=False), ex=ttl_seconds
            )
            await self._redis.sadd(index_key, entry_id)
            await self._redis.expire(index_key, ttl_seconds)
        except Exception as exc:
            print(f"[semcache-redis] store failed: {exc}")
            return None
        return entry_id


_singleton: RedisSemanticCache | None = None


def get_redis_semantic_cache() -> RedisSemanticCache:
    global _singleton
    if _singleton is None:
        _singleton = RedisSemanticCache()
    return _singleton
