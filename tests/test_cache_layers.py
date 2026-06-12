from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock

import pytest

from vietjet.cache import (
    compute_context_hash,
    is_realtime_intent,
    normalize_query,
    stable_hash,
)


def test_stable_hash_deterministic():
    assert stable_hash("hello") == stable_hash("hello")
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
    assert stable_hash("abc") != stable_hash("abd")


def test_normalize_query_picks_slots():
    out = normalize_query("phí đổi vé Eco VietJet nội địa?")
    assert out["slots"]["ticket_class"] == "eco"
    assert out["slots"]["airline"] == "vietjet"
    assert out["slots"]["route_type"] == "dom"
    assert out["slots"]["group_policy"] == "ticket_change"


def test_normalize_query_collapses_synonyms():
    a = normalize_query("phi doi ve eco vj noi dia")
    b = normalize_query("phí đổi vé Eco VietJet nội địa?")
    assert a["normalized_query"].split(" | ")[0] == b["normalized_query"].split(" | ")[0]


def test_normalize_query_strips_đ():
    out = normalize_query("đổi địa điểm")
    assert "đ" not in out["normalized_query"]
    assert "doi" in out["normalized_query"]


def test_realtime_intent_detects_keywords():
    assert is_realtime_intent("giá vé hôm nay bao nhiêu")
    assert is_realtime_intent("trạng thái chuyến bay hiện tại")
    assert is_realtime_intent("status booking")
    assert not is_realtime_intent("điều kiện vé eco là gì")
    assert not is_realtime_intent("phí ký gửi hành lý")


def test_compute_context_hash_stable():
    a = compute_context_hash([("doc_1", "v1"), ("doc_2", "v2")])
    b = compute_context_hash([("doc_1", "v1"), ("doc_2", "v2")])
    c = compute_context_hash([("doc_2", "v2"), ("doc_1", "v1")])
    assert a == b
    assert a != c


@pytest.mark.asyncio
async def test_embedding_cache_round_trip(monkeypatch):
    from vietjet.cache.layers import get_query_embedding

    calls = {"n": 0}

    async def fake_embed(text: str) -> list[float]:
        calls["n"] += 1
        return [0.1, 0.2, 0.3]

    class FakeStore:
        def __init__(self):
            self.data: dict[str, dict] = {}

        def build_key(self, *parts):
            return ":".join(parts)

        async def get_json(self, key):
            return self.data.get(key)

        async def set_json(self, key, value, ttl_seconds):
            self.data[key] = value
            return True

    store = FakeStore()
    v1 = await get_query_embedding(store, "hello", fake_embed)
    v2 = await get_query_embedding(store, "hello", fake_embed)
    assert v1 == v2 == [0.1, 0.2, 0.3]
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_final_answer_cache_round_trip(monkeypatch):
    from vietjet.cache.layers import get_final_answer, set_final_answer

    class FakeStore:
        def __init__(self):
            self.data: dict[str, dict] = {}

        def build_key(self, *parts):
            return ":".join(parts)

        async def get_json(self, key):
            return self.data.get(key)

        async def set_json(self, key, value, ttl_seconds):
            self.data[key] = value
            return True

    store = FakeStore()
    norm = "vietjet eco ticket_change"
    payload = {
        "answer": "Phí 350000 VND",
        "context_hash": "ctx_1",
        "sources": [],
    }
    await set_final_answer(store, "tenant_a", "public", norm, payload)
    hit = await get_final_answer(store, "tenant_a", "public", norm)
    assert hit is not None
    assert hit["answer"] == "Phí 350000 VND"

    miss = await get_final_answer(
        store, "tenant_a", "public", norm, expected_context_hash="ctx_2"
    )
    assert miss is None
