from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

import redis.asyncio as aioredis

from vietjet.config import REDIS_KEY_PREFIX, REDIS_URL


def stable_hash(data: Any) -> str:
    if not isinstance(data, str):
        data = json.dumps(data, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:24]


class CacheStore:
    def __init__(self, redis_client: aioredis.Redis, prefix: str = REDIS_KEY_PREFIX):
        self.redis = redis_client
        self.prefix = prefix

    def build_key(self, *parts: str) -> str:
        return ":".join([self.prefix, *parts])

    async def get_json(self, key: str) -> Optional[dict]:
        try:
            raw = await self.redis.get(key)
        except Exception as exc:
            print(f"[cache] get_json failed key={key} err={exc}")
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def set_json(self, key: str, value: dict, ttl_seconds: int) -> bool:
        payload = json.dumps(value, ensure_ascii=False)
        try:
            await self.redis.set(key, payload, ex=ttl_seconds)
            return True
        except Exception as exc:
            print(f"[cache] set_json failed key={key} err={exc}")
            return False

    async def delete(self, key: str) -> int:
        try:
            return int(await self.redis.delete(key))
        except Exception as exc:
            print(f"[cache] delete failed key={key} err={exc}")
            return 0

    async def exists(self, key: str) -> bool:
        try:
            return bool(await self.redis.exists(key))
        except Exception:
            return False

    async def ping(self) -> bool:
        try:
            return bool(await self.redis.ping())
        except Exception:
            return False

    async def close(self) -> None:
        try:
            await self.redis.aclose()
        except Exception:
            pass


_singleton: CacheStore | None = None


def get_cache_store() -> CacheStore:
    global _singleton
    if _singleton is None:
        client = aioredis.from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=2.0,
            socket_timeout=2.0,
        )
        _singleton = CacheStore(client, prefix=REDIS_KEY_PREFIX)
    return _singleton
