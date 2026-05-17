"""Vietjet RAG configuration.

Centralized constants for the full pipeline + agent. Override via env.
"""

from __future__ import annotations
import os
from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "raw_data"
CLEAN_DIR = ROOT_DIR / "clean_data"
NORM_DIR = ROOT_DIR / "normalized_data"
INDEX_DIR = ROOT_DIR / "index"
CHUNKS_PATH = ROOT_DIR / "chunks.jsonl"
BM25_PATH = INDEX_DIR / "bm25.pkl"

# --- Firecrawl ---
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "fc-ef65ae4a8aa2443791202dd2d113e72d")
CRAWL_MAX_RETRIES = 3
CRAWL_RETRY_BACKOFF = 2.0

# --- Postgres + pgvector ---
DB_CONNECTION_STRING = os.getenv(
    "DATABASE_URL", "postgresql://admin:123456@localhost:5432/rag_db"
)
COLLECTION_NAME = os.getenv("VIETJET_COLLECTION", "vietjet_collection")

# --- Models ---
# IDs are the HF Hub repo ids; the resolved EMBED_MODEL/RERANK_MODEL below
# point to local snapshots in models/hf/ when available (downloaded via
# `python -m vietjet.download_models`), avoiding any HF Hub call.
EMBED_MODEL_ID = os.getenv("VIETJET_EMBED_MODEL", "bkai-foundation-models/vietnamese-bi-encoder")
RERANK_MODEL_ID = os.getenv("VIETJET_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
LLM_MODEL = os.getenv("VIETJET_LLM_MODEL", "qwen2.5")

from vietjet.models import resolve  # noqa: E402  (import after env-defaults)

EMBED_MODEL = resolve(EMBED_MODEL_ID)
RERANK_MODEL = resolve(RERANK_MODEL_ID)

# --- Retrieval ---
RRF_K = 60
TOP_K = 4
CANDIDATES = 20
RERANK_MAX_LENGTH = 512
RERANK_TEXT_CAP = 2000  # truncate chunk text fed to reranker to dodge MPS OOM

# --- Agent ---
MAX_REWRITES = 2
GRADE_THRESHOLD = 0.5  # heuristic fraction of "grounded" tokens

# --- URLs to crawl ---
URLS: list[dict[str, str]] = [
    {"filename": "dieu-le-van-chuyen-vietjet", "url": "https://www.vietjetair.com/vi/pages/dieu-le-van-chuyen-vietjet-1618221808366"},
    {"filename": "dieu-le-van-chuyen-vietjet-thai-lan", "url": "https://www.vietjetair.com/vi/pages/dieu-le-van-chuyen-vietjet-thai-lan-1714099138124"},
    {"filename": "dieu-kien-ve", "url": "https://www.vietjetair.com/vi/pages/de-co-chuyen-bay-tot-dep-1578323501979/dieu-kien-ve-1641466500765"},
    {"filename": "dieu-kien-ve-chang-bay-lien-danh-giua-vietjet-va-lao-airlines", "url": "https://www.vietjetair.com/vi/pages/dieu-kien-ve-chang-bay-lien-danh-giua-vietjet-va-lao-airlines-1730099285032"},
    {"filename": "thong-tin-boi-thuong", "url": "https://www.vietjetair.com/vi/pages/de-co-chuyen-bay-tot-dep-1578323501979/thong-tin-boi-thuong-1578483460118"},
    {"filename": "phi-va-le-phi", "url": "https://www.vietjetair.com/vi/pages/de-co-chuyen-bay-tot-dep-1578323501979/phi-va-le-phi-1578483039924"},
    {"filename": "giay-to-tuy-than", "url": "https://www.vietjetair.com/vi/pages/de-co-chuyen-bay-tot-dep-1578323501979/giay-to-tuy-than-1578483122906"},
    {"filename": "san-bay-va-nha-ga-quoc-te", "url": "https://www.vietjetair.com/vi/pages/de-co-chuyen-bay-tot-dep-1578323501979/san-bay-va-nha-ga-quoc-te-1578483188498"},
    {"filename": "quy-dinh-hanh-ly", "url": "https://www.vietjetair.com/vi/pages/de-co-chuyen-bay-tot-dep-1578323501979/quy-dinh-hanh-ly-1578483259803"},
    {"filename": "tim-kiem-hanh-ly", "url": "https://www.vietjetair.com/vi/pages/de-co-chuyen-bay-tot-dep-1578323501979/tim-kiem-hanh-ly-1578483293321"},
    {"filename": "thong-tin-noi-chuyen", "url": "https://www.vietjetair.com/vi/pages/de-co-chuyen-bay-tot-dep-1578323501979/thong-tin-noi-chuyen-1578483387074"},
    {"filename": "kenh-thanh-toan", "url": "https://www.vietjetair.com/vi/pages/de-co-chuyen-bay-tot-dep-1578323501979/kenh-thanh-toan-1578483531654"},
    {"filename": "hoa-don-vat", "url": "https://www.vietjetair.com/vi/pages/de-co-chuyen-bay-tot-dep-1578323501979/hoa-don-vat-1599449490635"},
    {"filename": "huong-dan-lam-thu-tuc-chuyen-bay", "url": "https://www.vietjetair.com/vi/pages/de-co-chuyen-bay-tot-dep-1578323501979/huong-dan-lam-thu-tuc-chuyen-bay-1685509950583"},
    {"filename": "thu-cung-oi-bay-thoi", "url": "https://www.vietjetair.com/vi/pages/sky-pet---thu-cung-oi-bay-thoi-1717385172883"},
    {"filename": "tre-tu-tin-bay-mot-minh", "url": "https://www.vietjetair.com/vi/pages/sky-kids---tre-tu-tin-bay-mot-minh-1717385418700"},
    {"filename": "dich-vu-ho-tro-bay-cung-ban", "url": "https://www.vietjetair.com/vi/pages/mua-hanh-ly-suat-an-chon-cho-ngoi-va-hon-the-nua-1754713926921/dich-vu-ho-tro-bay-cung-ban-1719195650512"},
    {"filename": "hang-ve-thuong-gia---business", "url": "https://www.vietjetair.com/vi/pages/dich-vu-cao-cap-1689909996583/hang-ve-thuong-gia---business-1689909772703"},
    {"filename": "hang-ve-skyboss", "url": "https://www.vietjetair.com/vi/pages/dich-vu-cao-cap-1689909996583/hang-ve-skyboss-1689909971054"},
    {"filename": "phong-cho-sang-trong", "url": "https://www.vietjetair.com/vi/pages/dich-vu-cao-cap-1689909996583/phong-cho-sang-trong-1578484208407"},
]

DOC_TYPE_MAP: dict[str, str] = {
    "dieu-le-van-chuyen-vietjet": "regulation",
    "dieu-le-van-chuyen-vietjet-thai-lan": "regulation",
    "dieu-kien-ve": "regulation",
    "dieu-kien-ve-chang-bay-lien-danh-giua-vietjet-va-lao-airlines": "regulation",
    "thong-tin-boi-thuong": "compensation",
    "phi-va-le-phi": "pricing",
    "hoa-don-vat": "pricing",
    "quy-dinh-hanh-ly": "baggage",
    "tim-kiem-hanh-ly": "baggage",
    "giay-to-tuy-than": "procedure",
    "huong-dan-lam-thu-tuc-chuyen-bay": "procedure",
    "thong-tin-noi-chuyen": "procedure",
    "kenh-thanh-toan": "payment",
    "thu-cung-oi-bay-thoi": "service",
    "tre-tu-tin-bay-mot-minh": "service",
    "dich-vu-ho-tro-bay-cung-ban": "service",
    "hang-ve-thuong-gia---business": "service",
    "hang-ve-skyboss": "service",
    "phong-cho-sang-trong": "service",
    "san-bay-va-nha-ga-quoc-te": "procedure",
}
