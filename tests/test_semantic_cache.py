from __future__ import annotations

import random

import pytest

from vietjet.cache import ensure_schema, get_semantic_cache


def _try_schema() -> bool:
    try:
        ensure_schema()
        return True
    except Exception as exc:
        print(f"semantic-cache schema not available: {exc}")
        return False


@pytest.fixture(scope="module")
def db_available() -> bool:
    return _try_schema()


def test_semantic_cache_hit_when_same_embedding(db_available):
    if not db_available:
        pytest.skip("postgres+pgvector not available")
    sc = get_semantic_cache()
    random.seed(7)
    emb = [random.random() for _ in range(768)]

    sc.store(
        question="test q",
        normalized_query="test_norm_unique_xyz",
        answer="A",
        sources=[{"source": "s1"}],
        context_hash="ctx_a",
        embedding=emb,
    )
    hit = sc.lookup(emb)
    assert hit is not None
    assert hit.context_hash == "ctx_a"


def test_semantic_cache_context_hash_mismatch_misses(db_available):
    if not db_available:
        pytest.skip("postgres+pgvector not available")
    sc = get_semantic_cache()
    random.seed(11)
    emb = [random.random() for _ in range(768)]

    sc.store(
        question="ctx test",
        normalized_query="ctx_mismatch_norm_q",
        answer="answer",
        sources=[],
        context_hash="ctx_v1",
        embedding=emb,
    )
    miss = sc.lookup(emb, expected_context_hash="ctx_v2")
    assert miss is None
    hit = sc.lookup(emb, expected_context_hash="ctx_v1")
    assert hit is not None
