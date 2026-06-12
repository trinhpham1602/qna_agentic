# Plan Tổng Hợp: Agentic RAG với Multi-Layer Cache + Parallel Web Crawl + Streaming

> Mục tiêu: 1 plan duy nhất để triển khai Agentic RAG hoàn chỉnh từ skeleton đến production.

---

## 1. Mục tiêu hệ thống

Xây Agentic RAG (Vietjet domain) với 4 đặc tính:

1. **Đa tầng cache** — giảm latency, giảm cost LLM/embed/web API
2. **Parallel retrieval** — DB pgvector + Web search chạy song song
3. **Streaming early-answer** — trả lời sớm khi đủ confidence, không đợi crawl xong
4. **Background ingest** — page crawl dư được upsert vào pgvector cho lần sau

Công thức tối ưu:

```text
Latency  = cache nhiều tầng + parallel retrieve + timeout ngắn
Cost     = embedding cache + retrieval cache + web page cache
Correct  = context_hash + version-based invalidation
Scale    = semantic cache + hot key tracking + background ingest
```

---

## 2. Kiến trúc tổng thể

```text
                              User Question
                                    │
                                    ▼
                          ┌───────────────────┐
                          │  normalize_query  │  (typo, viết tắt, không dấu)
                          └─────────┬─────────┘
                                    │
                          ┌─────────▼──────────┐
                          │ check_semantic_    │  (pgvector similarity ≥ 0.93
                          │ answer_cache       │   + context_hash valid
                          │                    │   + skip realtime intent)
                          └─────────┬──────────┘
                              hit │   │ miss
                                  ▼   │
                          return cache │
                                       ▼
                          ┌────────────────────┐
                          │ check_final_       │  (exact normalized_query
                          │ answer_cache       │   + context_hash valid)
                          └─────────┬──────────┘
                              hit │   │ miss
                                  ▼   │
                          return cache │
                                       ▼
                          ┌────────────────────┐
                          │  embedding_cache   │  (model + normalized_query)
                          └─────────┬──────────┘
                                    │
                          ┌─────────▼──────────┐
                          │  cache_check_db    │  (URL trong pgvector
                          │  (parallel-crawl   │   < 1h AND sim ≥ thr)
                          │   cache layer)     │
                          └─────────┬──────────┘
                              hit │   │ miss
                                  ▼   │
                          fast_path:   │
                          db_retrieve  │
                          → generate   ▼
                                ┌─────────────────────────────────┐
                                │  parallel fan-out               │
                                └──┬───────────────────────────┬──┘
                                   │                           │
                                   ▼                           ▼
                          ┌────────────────┐         ┌──────────────────┐
                          │ db_retrieve    │         │ crawl_coordinator│
                          │ (pgvector +    │         │ (N sub-agents    │
                          │  BM25 + rerank │         │  + Firecrawl     │
                          │ retrieval_cache│         │  watcher stream) │
                          └────────┬───────┘         └─────────┬────────┘
                                   │                           │
                                   │              ┌────────────┴────────────┐
                                   │              │                          │
                                   │              ▼                          ▼
                                   │      ┌──────────────┐         ┌──────────────┐
                                   │      │ judge_       │         │ URL frontier │
                                   │      │ consumer     │◀────────│ + seen set   │
                                   │      │ (LLM + embed │         │  (dedup)     │
                                   │      │  sim)        │         └──────────────┘
                                   │      └──────┬───────┘
                                   │             │ match (sim≥0.75 AND conf=high)
                                   │             ▼
                                   │      ┌──────────────┐
                                   │      │ fire EARLY   │ ──▶ SSE event "partial"
                                   │      │ ANSWER       │
                                   │      │ + switch     │
                                   │      │  mode=bg     │
                                   │      └──────┬───────┘
                                   │             │
                                   │             ▼
                                   │      ┌──────────────────────────┐
                                   │      │ background_ingest        │
                                   │      │ (clean→chunk→embed→      │
                                   │      │  upsert pgvector)        │
                                   │      └──────────────────────────┘
                                   │             │
                                   └─────────────┴──────┐
                                                        ▼
                                              ┌────────────────┐
                                              │  grade_context │
                                              └────────┬───────┘
                                              enough │  │ missing
                                                     │  └─────▶ rewrite ↺
                                                     ▼
                                              ┌────────────────┐
                                              │  merge_context │
                                              │  (DB + web,    │
                                              │   rerank top-K)│
                                              └────────┬───────┘
                                                       ▼
                                              ┌────────────────┐
                                              │ generate_answer│
                                              │ (cite DB + web)│
                                              └────────┬───────┘
                                                       ▼
                                              ┌────────────────┐
                                              │ store cache    │
                                              │ (answer +      │
                                              │  semantic +    │
                                              │  context_hash) │
                                              └────────┬───────┘
                                                       ▼
                                                 Final Answer
                                                 (SSE event "done")
```

---

## 3. Các tầng cache (từ Plan 1)

### 3.1. Bảng tóm tắt tầng cache

| Tầng | Mục đích | Key | TTL gợi ý |
|------|---------|-----|-----------|
| **Normalize** | Map nhiều câu hỏi về dạng chuẩn | `norm:{hash(raw)}` | 7–30 ngày |
| **Embedding** | Không gọi embed model nhiều lần | `emb:{model}:{hash(norm)}` | 30–90 ngày |
| **Retrieval** | Không query vector DB lặp | `retrieval:{coll}:{ver}:{hash}` | 1h–7 ngày |
| **Document** | Cache content doc theo version | `doc:{doc_id}:{ver}` | 1–30 ngày |
| **Chunk** | Tránh chunk lại | `chunk:{chunk_id}:{ver}` | 7–30 ngày |
| **Web search** | Cache kết quả search API | `websearch:{provider}:{hash}` | 5p–1 ngày |
| **Web page** | Cache content URL đã crawl | `webpage:{hash(url)}` | 15p–7 ngày |
| **Semantic answer** | Reuse answer cho câu tương tự | pgvector `semantic_answer_cache` | 15p–1 ngày |
| **Final answer** | Exact match | `answer:{tenant}:{scope}:{hash}` | 5p–1 ngày |

### 3.2. Naming convention

```text
{layer}:{domain}:{version}:{hash}
```

Ví dụ:

```text
norm:airline:v1:abc123
emb:airline:bge-m3:v1:abc123
retrieval:airline_policy:v4:abc123
websearch:tavily:v1:abc123
webpage:v1:def456
answer:airline:v2:abc123
```

### 3.3. Hash helper

```python
import hashlib, json
from typing import Any

def stable_hash(data: Any) -> str:
    if not isinstance(data, str):
        data = json.dumps(data, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:24]
```

### 3.4. Semantic answer cache (pgvector)

```sql
CREATE TABLE semantic_answer_cache (
    id BIGSERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    user_scope TEXT NOT NULL DEFAULT 'public',
    normalized_query TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources JSONB NOT NULL,
    context_hash TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP NOT NULL
);

CREATE INDEX semantic_answer_cache_embedding_idx
ON semantic_answer_cache
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

Search:

```sql
SELECT id, question, answer, sources, context_hash,
       1 - (embedding <=> :q_embed) AS similarity
FROM semantic_answer_cache
WHERE tenant_id = :tid AND user_scope = :scope
  AND expires_at > NOW()
ORDER BY embedding <=> :q_embed
LIMIT 3;
```

Use cache nếu `similarity ≥ 0.93 AND context_hash valid AND query không realtime intent`.

### 3.5. Cache invalidation

3 chiến lược, dùng kết hợp:

| Chiến lược | Khi áp dụng |
|-----------|-------------|
| **Version-based** | Đổi retriever/chunker/embedding/prompt → bump version, không xóa key cũ |
| **Context hash** | Final answer cache lưu `context_hash` của docs; docs đổi → context_hash đổi → cache miss |
| **Event-based** | Admin update policy → publish event → invalidate doc/chunk + bump collection version |

### 3.6. Không cache (hoặc TTL cực ngắn)

```text
- Dữ liệu cá nhân user / booking status / payment status
- Giá vé realtime / flight status
- Câu hỏi chứa "mới nhất", "hôm nay", "hiện tại"
- Query phụ thuộc permission
```

Nếu cần cache user-specific: `answer:{tenant}:{user_id}:{hash}` — không public.

---

## 4. Parallel retrieval & Web search agent (từ Plan 2 + 3)

### 4.1. Hai chế độ vận hành

| Chế độ | Khi dùng | Flow |
|--------|---------|------|
| **Fast mode** | Chatbot realtime, query phổ biến | Cache → RAG → Web snippet → Generate (1 LLM call) |
| **Deep mode** | User hỏi phân tích kỹ | Cache → RAG → Parallel crawl → Judge stream → Rerank → Generate → Fact check |

Config khác biệt:

```python
# Fast
MAX_RAG_DOCS = 5
MAX_WEB_RESULTS = 3
EXTRACT_FULL_PAGE = False
WEB_TIMEOUT = 3.0

# Deep
MAX_RAG_DOCS = 10
MAX_WEB_RESULTS = 10
EXTRACT_FULL_PAGE = True
MAX_PAGES_EXTRACT = 3
WEB_TIMEOUT = 10.0
```

### 4.2. Sub-component cho parallel crawl

| Thành phần | Vai trò | File |
|-----------|---------|------|
| `URLFrontier` | `asyncio.Queue` + `seen: set[str]` dedup | `vietjet/crawl_parallel/frontier.py` |
| `CrawlAgent` | Pull URL → Firecrawl scrape → push vào judge/ingest queue | `vietjet/crawl_parallel/agent.py` |
| `JudgeConsumer` | LLM + embed sim → fire early-answer khi match | `vietjet/crawl_parallel/judge.py` |
| `BackgroundIngest` | Consume ingest queue → clean → chunk → embed → upsert | `vietjet/crawl_parallel/background.py` |
| `CrawlCoordinator` | Orchestrator: spawn agents, expose `aiter_events()` cho server | `vietjet/crawl_parallel/coordinator.py` |
| `CacheChecker` | Query pgvector với TTL filter | `vietjet/crawl_parallel/cache.py` |

### 4.3. Firecrawl streaming

Dùng `AsyncFirecrawlApp.crawl_url_and_watch(url, limit, max_depth, scrape_options)` — async iterator yield page-by-page ngay khi mỗi page xong.

Fallback: `start_crawl` + `get_crawl_status` polling nếu WebSocket lỗi.

### 4.4. Judge logic — early-stop signal

Không dùng "% match" literal. Thay bằng combined:

```python
THRESHOLD_HIGH_SIM = 0.75    # embedding cosine
THRESHOLD_MED_SIM = 0.60     # tối thiểu để gọi LLM rate

# Fire early-answer khi:
confidence == "high" AND sim >= THRESHOLD_HIGH_SIM
```

Judge prompt:

```text
Bạn đánh giá đoạn web có chứa câu trả lời cho câu hỏi không.
CÂU HỎI: {query}
ĐOẠN: {snippet}
Trả về CHỈ JSON: {"confidence": "high"|"medium"|"low", "reason": "<1 câu>"}
```

### 4.5. Lifecycle 1 query (deep mode)

```text
t=0:    Request đến /qa-stream (SSE)
t=10ms: CacheChecker chạy DB query
        - hit (URL <1h AND sim≥thr) → answer luôn, END
        - miss → coordinator start
t=10ms: Frontier seed home URLs
        Firecrawl crawl_url_and_watch với limit, max_depth=2
t=2-8s: Page đầu tiên về → JudgeConsumer chấm
t=4-12s: 1 page đạt match=high → fire SSE "partial_answer"
        → mode = "background"
t=after: CrawlAgent đẩy page còn lại vào ingest_queue
         BackgroundIngest upsert pgvector với metadata
         (last_crawled_at, source_query, session_id)
t=15s:  EARLY_ANSWER_TIMEOUT — nếu chưa match, emit best-effort
t=60s:  MAX_TASK_LIFETIME — cancel background tasks → "done"
```

### 4.6. pgvector metadata bổ sung

| Field | Kiểu | Mục đích |
|-------|------|---------|
| `last_crawled_at` | ISO datetime | Cache TTL check |
| `source_query` | str (nullable) | Query đã trigger crawl URL này |
| `crawl_session_id` | str (nullable) | Trace 1 lượt crawl |
| `doc_type` | `"web_live"` \| `"seed"` | Phân biệt nguồn |

PGVector lưu metadata trong JSONB → không cần ALTER TABLE, chỉ update ingest code.

---

## 5. AgentState (LangGraph)

```python
from typing import TypedDict, List, Optional, Literal
from langchain_core.documents import Document

class AgentState(TypedDict, total=False):
    # Query
    question: str
    normalized_query: str
    query_embedding: list[float]

    # Routing
    route: Literal["rag_only", "web_only", "both", "cache"]
    intent_realtime: bool

    # Cache
    cache_hit: bool
    cache_layer_hit: Optional[str]   # "semantic" | "final" | "db_check"
    cached_answer: Optional[str]

    # Retrieve
    rag_docs: List[Document]
    web_docs: List[Document]
    web_candidates: List[dict]       # [{"url","title","snippet"}]
    web_chosen_urls: List[str]
    web_skipped_reason: Optional[str]

    # Crawl session
    crawl_session_id: str
    early_answer_emitted: bool

    # Merge & generate
    merged_context: str
    context_hash: str
    answer: str
    sources: List[dict]
    rewrites_remaining: int
```

---

## 6. Config tổng (`vietjet/config.py`)

```python
# --- Cache TTL (seconds) ---
TTL_NORMALIZE = 2_592_000          # 30 days
TTL_EMBEDDING = 5_184_000          # 60 days
TTL_RETRIEVAL = 86_400             # 1 day
TTL_DOC = 2_592_000                # 30 days
TTL_CHUNK = 2_592_000              # 30 days
TTL_WEB_SEARCH_LATEST = 900        # 15 min
TTL_WEB_SEARCH_NORMAL = 86_400     # 1 day
TTL_WEB_PAGE_OFFICIAL = 604_800    # 7 days
TTL_WEB_PAGE_NEWS = 1_800          # 30 min
TTL_WEB_PAGE_PRICE = 60            # 1 min
TTL_SEMANTIC_ANSWER = 86_400       # 1 day
TTL_FINAL_ANSWER = 3_600           # 1 hour

# --- Versioning ---
COLLECTION_VERSION = "airline-policy-v4"
RETRIEVER_VERSION = "v3"
CHUNKER_VERSION = "v2"
EMBEDDING_MODEL_VERSION = "bkai-bi-encoder-v1"
PROMPT_VERSION = "answer-v5"

# --- Semantic cache ---
SEMANTIC_CACHE_ENABLED = True
SEMANTIC_CACHE_THRESHOLD = 0.93
SEMANTIC_CACHE_SKIP_REALTIME = True

# --- Web search ---
WEB_SEARCH_HOMES = ["https://www.vietjetair.com/vi"]
WEB_SEARCH_LIMIT = 8
WEB_SEARCH_TIMEOUT = 8.0
WEB_JUDGE_MAX_PICKS = 2
WEB_FETCH_CHUNK_CAP = 4000
WEB_FETCH_TIMEOUT = 10.0

# --- Parallel crawl ---
PARALLEL_CRAWL_HOMES = ["https://www.vietjetair.com/vi"]
MAX_CONCURRENT_AGENTS = 4
MAX_PAGES_PER_QUERY = 30
EARLY_ANSWER_TIMEOUT = 15.0
MAX_TASK_LIFETIME = 60.0
CACHE_TTL_SECONDS = 3600
CACHE_SIM_THRESHOLD = 0.70
JUDGE_SIM_HIGH = 0.75
JUDGE_SIM_MED = 0.60
INGEST_BATCH_SIZE = 5
MAX_REWRITES = 1
```

---

## 7. LangGraph nodes & edges

### 7.1. Node list

| Node | Vai trò |
|------|---------|
| `normalize_query` | Chuẩn hóa câu hỏi, slot extract |
| `check_semantic_cache` | pgvector similarity ≥ threshold + context_hash |
| `check_final_cache` | Exact match normalized_query |
| `get_query_embedding` | embedding_cache → embed model |
| `route` | Decide cache / rag_only / both / web_only |
| `db_retrieve` | retrieval_cache → pgvector + BM25 + rerank |
| `crawl_stream` | Spawn `CrawlCoordinator` (parallel agents + judge) |
| `link_judge` | (Fast mode only) LLM chọn 1-2 URL từ candidates |
| `web_fetch` | (Fast mode) scrape các URL chosen |
| `merge` | Gộp rag_docs + web_docs |
| `grade_context` | Đánh giá đủ context chưa |
| `rewrite_query` | Reformulate khi grade fail |
| `generate` | Sinh answer với citation |
| `store_cache` | Lưu answer + semantic cache + context_hash |
| `return_cached_answer` | Short-circuit khi cache hit |

### 7.2. Edges

```python
g.add_node("normalize_query", normalize_query_node)
g.add_node("check_semantic_cache", check_semantic_cache_node)
g.add_node("check_final_cache", check_final_cache_node)
g.add_node("get_query_embedding", get_query_embedding_node)
g.add_node("db_retrieve", db_retrieve_node)
g.add_node("crawl_stream", crawl_stream_node)
g.add_node("merge", merge_node)
g.add_node("grade_context", grade_context_node)
g.add_node("rewrite_query", rewrite_query_node)
g.add_node("generate", generate_node)
g.add_node("store_cache", store_cache_node)
g.add_node("return_cached_answer", return_cached_node)

g.set_entry_point("normalize_query")
g.add_edge("normalize_query", "check_semantic_cache")

g.add_conditional_edges(
    "check_semantic_cache",
    lambda s: "return_cached_answer" if s["cache_hit"] else "check_final_cache",
)
g.add_conditional_edges(
    "check_final_cache",
    lambda s: "return_cached_answer" if s["cache_hit"] else "get_query_embedding",
)

g.add_edge("get_query_embedding", "db_retrieve")
g.add_edge("get_query_embedding", "crawl_stream")     # fan-out parallel

g.add_edge("db_retrieve", "merge")
g.add_edge("crawl_stream", "merge")                    # merge chờ cả 2

g.add_edge("merge", "grade_context")
g.add_conditional_edges(
    "grade_context",
    lambda s: "rewrite_query" if (not s["enough"] and s["rewrites_remaining"] > 0)
              else "generate",
)
g.add_edge("rewrite_query", "db_retrieve")             # loop
g.add_edge("generate", "store_cache")
g.add_edge("store_cache", END)
g.add_edge("return_cached_answer", END)
```

Parallel merge: dùng reducer `operator.add` cho field `docs` và `web_docs`, hoặc `Send` API.

---

## 8. Files cần tạo / sửa

### 8.1. NEW

| File | Mô tả |
|------|------|
| `vietjet/cache/__init__.py` | Module init |
| `vietjet/cache/store.py` | `CacheStore` (Redis-like, có thể dùng PostgreSQL backend) |
| `vietjet/cache/semantic.py` | `SemanticAnswerCache` (pgvector) |
| `vietjet/cache/normalize.py` | Query normalize + slot extract |
| `vietjet/crawl_parallel/__init__.py` | Module init |
| `vietjet/crawl_parallel/frontier.py` | `URLFrontier` |
| `vietjet/crawl_parallel/agent.py` | `CrawlAgent` |
| `vietjet/crawl_parallel/judge.py` | `JudgeConsumer` |
| `vietjet/crawl_parallel/background.py` | `BackgroundIngest` |
| `vietjet/crawl_parallel/coordinator.py` | `CrawlCoordinator` |
| `vietjet/crawl_parallel/cache.py` | `CacheChecker` (DB TTL filter) |
| `vietjet/web_search.py` | Firecrawl search wrapper (fast mode) |
| `vietjet/judge.py` | Link judge LLM (fast mode) |
| `tests/test_cache_layers.py` | Unit test cache layer |
| `tests/test_parallel_crawl.py` | Unit test frontier/judge/coordinator |
| `tests/test_semantic_cache.py` | Semantic cache TTL hit/miss |

### 8.2. EDIT

| File | Sửa gì |
|------|--------|
| `vietjet/config.py` | Thêm constants ở §6 |
| `vietjet/agent.py` | Rebuild graph với nodes ở §7 |
| `vietjet/retriever.py` | Thêm `search_with_metadata_filter(last_crawled_at__gte=...)` |
| `vietjet/ingest.py` | Hàm `upsert_pages(pages, metadata)` cho background path |
| `vietjet/server.py` | Endpoint `/qa-stream` SSE |

---

## 9. Server SSE endpoint

```python
@app.post("/qa-stream")
async def qa_stream(req: QueryRequest):
    async def event_gen():
        async for ev in coordinator.stream(req.query, PARALLEL_CRAWL_HOMES):
            yield f"event: {ev.type}\ndata: {json.dumps(ev.payload)}\n\n"
    return StreamingResponse(event_gen(), media_type="text/event-stream")
```

Event types FE handle:

| Event | Khi nào | FE action |
|-------|---------|-----------|
| `cache_hit` | Cache layer hit | Render full answer ngay |
| `partial_answer` | Judge fire early | Render answer (status "partial") |
| `ingested` | Background ingest 1 batch xong | Debug only |
| `final` | Generate xong full context | Replace partial nếu khác |
| `done` | Session kết thúc | Cleanup connection |

---

## 10. Thứ tự triển khai (chia 9 PR)

> Mỗi step 1 PR độc lập, merge tuần tự. Không nhảy bước.

### Step 1 — Foundation (offline)
- Tạo skeleton package `vietjet/cache/`, `vietjet/crawl_parallel/`
- Implement `URLFrontier` + unit test concurrent put/get + dedup
- Thêm config constants
- Implement `stable_hash` + `CacheStore.get_json/set_json`

### Step 2 — Cache layers (Redis hoặc Postgres backend)
- Normalize query helper + cache
- Embedding cache wrapper bọc embed model
- Retrieval cache wrapper bọc retriever
- Test: hit miss đúng, key có version

### Step 3 — Semantic answer cache
- Migration tạo bảng `semantic_answer_cache` + ivfflat index
- Implement `SemanticAnswerCache.lookup` + `.store`
- Realtime intent detector (keyword "mới nhất", "hôm nay", "giá", "status")
- Test: 2 câu cùng nghĩa → cache hit, câu realtime → skip

### Step 4 — Firecrawl streaming wrapper
- Implement `CrawlAgent` thuần (chưa judge/ingest)
- Smoke test: `crawl_url_and_watch` 1 home URL → in từng page
- Đo latency start → page đầu tiên

### Step 5 — Judge consumer
- `JudgeConsumer` với snippet extractor + embed sim + LLM rate
- Test với 3 mock page: chỉ page match được fire

### Step 6 — Background ingest
- `BackgroundIngest` reuse `vietjet.clean` + `vietjet.chunk` + `ingest`
- Sửa `ingest.py` thêm `upsert_pages` (không full rebuild)
- Verify metadata `last_crawled_at`, `source_query`, `crawl_session_id` đúng

### Step 7 — Cache check (DB TTL filter)
- `CacheChecker.check_db(query, embedding)` → filter pgvector `last_crawled_at >= now - TTL`
- Test: 1 doc cũ + 1 doc mới → chỉ doc mới trả về

### Step 8 — Coordinator + early-answer event
- Wire toàn bộ trong `CrawlCoordinator`
- Integration test: 1 home URL, 3 agent, 1 query match keyword → early-answer fire trong <15s
- Verify `mode = "background"` switch đúng, ingest queue được tiêu thụ

### Step 9 — Graph + Server SSE
- Rebuild `vietjet/agent.py` với nodes ở §7
- Endpoint `/qa-stream` SSE
- Test bằng `curl -N` xem event chunked đúng
- Backward-compat: `/qa` non-stream vẫn chạy (đợi `final` rồi flush 1 lần)

### Step 10 — Hardening (sau khi merge)
- Quota guard: đếm pages-crawled-today, refuse khi vượt
- Retry / fallback WebSocket → polling
- Metrics: log session_id, page count, judge confidence distribution
- Load test: 10 concurrent query, verify không race condition
- Daily cleanup job: expire pgvector docs > 30 ngày

---

## 11. Edge cases tổng hợp

| Trường hợp | Xử lý |
|-----------|------|
| Cache Redis/PG lỗi | Degrade gracefully, vẫn answer được (skip cache) |
| Firecrawl quota hết | Catch trong agent → log + drop task, agent khác tiếp tục |
| WebSocket disconnect | Fallback `start_crawl` + polling, retry 1 lần |
| Judge LLM JSON sai | Treat `confidence=low`, không fire early |
| Cache check trả docs nhưng `last_crawled_at` cũ | Vẫn dùng làm context cho generate, NHƯNG không skip web crawl |
| User cancel SSE | EventGenerator GC → coordinator cancel tasks qua `CancelledError` |
| Same query <1h | Cache hit → answer luôn, không spawn agent |
| Background ingest fail (DB down) | Log + KHÔNG fail user (early answer đã trả) |
| Embedding OOM khi batch | Giảm `INGEST_BATCH_SIZE` về 2 |
| LangGraph parallel race trên state | Dùng reducer thay vì direct assignment |
| Cache có context_hash cũ nhưng docs đã update | context_hash mismatch → cache miss → re-generate |
| Câu hỏi có "mới nhất"/"hiện tại" | Skip semantic cache + final cache, force web search |

---

## 12. Metrics cần đo

```text
query_cache_hit_rate
semantic_cache_hit_rate
embedding_cache_hit_rate
retrieval_cache_hit_rate
websearch_cache_hit_rate
webpage_cache_hit_rate
db_cache_check_hit_rate          (parallel-crawl layer)
judge_early_fire_rate
avg_latency_cache_hit
avg_latency_partial_answer       (TTFB của SSE)
avg_latency_final_answer
llm_calls_saved
web_calls_saved
cost_saved
stale_answer_count
background_ingest_queue_depth
judge_confidence_distribution
```

Target production:

| Metric | Target |
|--------|-------:|
| Embedding cache hit | > 70% |
| Retrieval cache hit | > 40% |
| Web page cache hit | > 50% |
| Semantic cache hit | > 20% |
| Latency cache hit | < 500ms |
| Latency partial answer | < 8s |
| Latency final answer | < 20s |
| Avg latency reduction vs no-cache | > 30% |

---

## 13. Production checklist

```text
[ ] Normalize query trước mọi cache layer
[ ] Mọi cache key có version (model/collection/retriever/prompt)
[ ] Final answer cache có context_hash
[ ] Semantic cache threshold ≥ 0.93
[ ] Realtime intent detector skip cache cho "mới nhất"/"giá"/"status"
[ ] User-specific data dùng key chứa user_id, không public
[ ] Web search TTL ngắn cho latest/news
[ ] Web page cache có ETag/Last-Modified nếu server hỗ trợ
[ ] Cache miss degrade gracefully (không fail)
[ ] Metric hit/miss từng tầng được log
[ ] Event invalidation khi admin update doc
[ ] Redis max memory + LRU eviction policy
[ ] Web search có timeout cứng
[ ] Background ingest fail không ảnh hưởng response user
[ ] Firecrawl quota daily cap + per-query cap
[ ] SSE disconnect → cancel coordinator tasks
[ ] MAX_CONCURRENT_AGENTS, MAX_PAGES_PER_QUERY, MAX_TASK_LIFETIME đều set
[ ] Judge dùng model nhẹ hơn generate (tiết kiệm cost)
[ ] pgvector upsert dùng ON CONFLICT (url) DO UPDATE
[ ] Snippet window ~500 char trước khi embed (giảm nhiễu)
[ ] Daily cleanup expire docs > 30 ngày
[ ] Load test ≥ 10 concurrent query verify không race
```

---

## 14. Open questions (xác nhận trước khi code Step 1)

1. **Cache backend**: PostgreSQL (như comment trong plan 1) hay Redis riêng? → PostgreSQL đỡ thêm dependency, nhưng Redis nhanh hơn cho TTL key-value.
2. **Crawl scope**: cho phép Firecrawl ra ngoài `vietjetair.com` không? → Đề xuất `allow_external_links=False`.
3. **Judge LLM**: dùng cùng `qwen2.5` hay model nhẹ riêng (`qwen2.5:1.5b`)? → Nhẹ riêng cho judge để tiết kiệm.
4. **Embedding cho judge**: cùng `bkai-foundation-models/vietnamese-bi-encoder` (768d) hay model khác? → Cùng, khỏi load thêm RAM.
5. **MAX_CONCURRENT_AGENTS = 4** có khớp quota Firecrawl không? → Tunable theo plan.
6. **Background ingest TTL**: docs > 30 ngày trong pgvector có cần expire? → Cleanup job riêng, không lo trong scope chính.
7. **Cache key user-specific**: tenant + user_id cho domain Vietjet có cần không? → Nếu có booking data per user thì có.
8. **Ưu tiên khi DB và Web mâu thuẫn**: DB nội bộ (chính sách chính thức) hay Web live? → Đề xuất DB ưu tiên, ghi chú rõ trong answer.
9. **Early-answer rerank**: khi fire early-answer, rerank top-5 từ union (DB + judge collected) trước generate? → Có — giữ chất lượng.

---

## 15. Xác nhận Open Questions

1. **Cache backend**: Redis riêng, localhost:6379
2. **Crawl scope**: Không cho phép Firecrawl ra ngoài `vietjetair.com`.
3. **Judge LLM**: dùng cùng `qwen2.5`.
4. **Embedding cho judge**: cùng `bkai-foundation-models/vietnamese-bi-encoder` (768d).
5. **MAX_CONCURRENT_AGENTS = 4** Tunable theo plan.
6. **Background ingest TTL**: Cleanup job riêng. Dùng redis thì để redis tự cleanup
7. **Cache key user-specific**: không cần.
8. **Ưu tiên khi DB và Web mâu thuẫn**: Web live.
9. **Early-answer rerank**: Có — giữ chất lượng.

---

**Kết thúc plan.** Sau khi user xác nhận các open question ở §14, bắt đầu Step 1.
