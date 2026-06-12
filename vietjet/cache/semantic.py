from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import psycopg
from psycopg.rows import dict_row

from vietjet.config import (
    DB_CONNECTION_STRING,
    SEMANTIC_CACHE_DIM,
    SEMANTIC_CACHE_TABLE,
    SEMANTIC_CACHE_TENANT,
    SEMANTIC_CACHE_THRESHOLD,
    TTL_SEMANTIC_ANSWER,
)


_DDL_TABLE = f"""
CREATE TABLE IF NOT EXISTS {SEMANTIC_CACHE_TABLE} (
    id BIGSERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    user_scope TEXT NOT NULL DEFAULT 'public',
    normalized_query TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources JSONB NOT NULL DEFAULT '[]'::jsonb,
    context_hash TEXT NOT NULL,
    embedding VECTOR({SEMANTIC_CACHE_DIM}) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP NOT NULL
);
"""

_DDL_INDEX_TENANT = f"""
CREATE INDEX IF NOT EXISTS {SEMANTIC_CACHE_TABLE}_tenant_scope_idx
ON {SEMANTIC_CACHE_TABLE} (tenant_id, user_scope, expires_at);
"""

_DDL_INDEX_EMB = f"""
CREATE INDEX IF NOT EXISTS {SEMANTIC_CACHE_TABLE}_embedding_idx
ON {SEMANTIC_CACHE_TABLE}
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
"""

_DDL_PGVECTOR_EXT = "CREATE EXTENSION IF NOT EXISTS vector;"


def ensure_schema(conn_str: str = DB_CONNECTION_STRING) -> None:
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(_DDL_PGVECTOR_EXT)
            cur.execute(_DDL_TABLE)
            cur.execute(_DDL_INDEX_TENANT)
            try:
                cur.execute(_DDL_INDEX_EMB)
            except psycopg.Error as exc:
                print(f"[semantic-cache] ivfflat index skipped: {exc}")
        conn.commit()


def _format_vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in embedding) + "]"


@dataclass
class SemanticHit:
    id: int
    question: str
    answer: str
    sources: list
    context_hash: str
    similarity: float
    normalized_query: str


class SemanticAnswerCache:
    def __init__(
        self,
        *,
        conn_str: str = DB_CONNECTION_STRING,
        table: str = SEMANTIC_CACHE_TABLE,
        tenant_id: str = SEMANTIC_CACHE_TENANT,
        threshold: float = SEMANTIC_CACHE_THRESHOLD,
    ) -> None:
        self.conn_str = conn_str
        self.table = table
        self.tenant_id = tenant_id
        self.threshold = threshold

    def lookup(
        self,
        embedding: list[float],
        *,
        user_scope: str = "public",
        expected_context_hash: Optional[str] = None,
        limit: int = 3,
    ) -> Optional[SemanticHit]:
        vec = _format_vector_literal(embedding)
        sql = f"""
            SELECT id, question, answer, sources, context_hash, normalized_query,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM {self.table}
            WHERE tenant_id = %s
              AND user_scope = %s
              AND expires_at > NOW()
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        try:
            with psycopg.connect(self.conn_str) as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(sql, (vec, self.tenant_id, user_scope, vec, limit))
                    rows = cur.fetchall()
        except psycopg.Error as exc:
            print(f"[semantic-cache] lookup failed: {exc}")
            return None

        for row in rows:
            sim = float(row["similarity"])
            if sim < self.threshold:
                break
            if expected_context_hash is not None and row["context_hash"] != expected_context_hash:
                continue
            sources = row["sources"]
            if isinstance(sources, str):
                try:
                    sources = json.loads(sources)
                except json.JSONDecodeError:
                    sources = []
            return SemanticHit(
                id=int(row["id"]),
                question=row["question"],
                answer=row["answer"],
                sources=sources or [],
                context_hash=row["context_hash"],
                similarity=sim,
                normalized_query=row["normalized_query"],
            )
        return None

    def store(
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
    ) -> Optional[int]:
        vec = _format_vector_literal(embedding)
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        sql = f"""
            INSERT INTO {self.table}
              (tenant_id, user_scope, normalized_query, question, answer,
               sources, context_hash, embedding, expires_at)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s::vector, %s)
            RETURNING id;
        """
        try:
            with psycopg.connect(self.conn_str) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql,
                        (
                            self.tenant_id,
                            user_scope,
                            normalized_query,
                            question,
                            answer,
                            json.dumps(sources, ensure_ascii=False),
                            context_hash,
                            vec,
                            expires_at,
                        ),
                    )
                    row = cur.fetchone()
                conn.commit()
        except psycopg.Error as exc:
            print(f"[semantic-cache] store failed: {exc}")
            return None
        return int(row[0]) if row else None

    def purge_expired(self) -> int:
        sql = f"DELETE FROM {self.table} WHERE expires_at <= NOW();"
        try:
            with psycopg.connect(self.conn_str) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    deleted = cur.rowcount
                conn.commit()
            return deleted
        except psycopg.Error as exc:
            print(f"[semantic-cache] purge failed: {exc}")
            return 0


_singleton: SemanticAnswerCache | None = None


def get_semantic_cache() -> SemanticAnswerCache:
    global _singleton
    if _singleton is None:
        _singleton = SemanticAnswerCache()
    return _singleton
