from __future__ import annotations

import asyncio

import pytest

from vietjet.cache_nodes import (
    _docs_to_context_seed,
    _doc_to_source,
    after_cache_check,
    normalize_node,
)


class FakeDoc:
    def __init__(self, metadata: dict):
        self.metadata = metadata


def test_after_cache_check_returns_cached_when_hit():
    assert after_cache_check({"cached_from": "semantic"}) == "return_cached"
    assert after_cache_check({"cached_from": "final"}) == "return_cached"


def test_after_cache_check_returns_route_when_miss():
    assert after_cache_check({"cached_from": None}) == "route"
    assert after_cache_check({}) == "route"


def test_docs_to_context_seed_extracts_ids():
    docs = [
        FakeDoc({"id": "doc_1", "last_crawled_ts": 1234.0}),
        FakeDoc({"id": "doc_2", "doc_type": "web_live"}),
        FakeDoc({"source": "url_3"}),
    ]
    seed = _docs_to_context_seed(docs)
    assert seed[0] == ("doc_1", 1234.0)
    assert seed[1] == ("doc_2", "web_live")
    assert seed[2] == ("url_3", "static")


def test_doc_to_source_extracts_metadata():
    d = FakeDoc({"source": "http://x", "section_path": "p", "doc_type": "web_live"})
    src = _doc_to_source(d)
    assert src["source"] == "http://x"
    assert src["url"] == "http://x"

    d2 = FakeDoc({"source": "internal", "doc_type": "regulation"})
    src2 = _doc_to_source(d2)
    assert src2["url"] is None


@pytest.mark.asyncio
async def test_normalize_node_extracts_slots():
    out = await normalize_node({"question": "phí đổi vé eco vietjet"})
    assert out["normalized_query"]
    assert out["slots"]["ticket_class"] == "eco"
    assert out["cached_from"] is None


@pytest.mark.asyncio
async def test_normalize_node_detects_realtime():
    out = await normalize_node({"question": "giá vé hôm nay là bao nhiêu"})
    assert out["intent_realtime"] is True


@pytest.mark.asyncio
async def test_normalize_node_non_realtime():
    out = await normalize_node({"question": "điều kiện vé eco bao gồm gì"})
    assert out["intent_realtime"] is False
