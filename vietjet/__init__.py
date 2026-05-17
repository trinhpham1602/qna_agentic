"""Vietjet RAG: crawl → clean → normalize → chunk → ingest → agentic Q&A.

Side effects on import (must run BEFORE any huggingface_hub / transformers
import elsewhere — that's why they live here in __init__.py):
  - Suppress noisy progress bars + telemetry.
  - If both required HF models are already snapshotted to models/hf/, flip
    HF_HUB_OFFLINE on so subsequent loads make zero network calls.
"""

from __future__ import annotations
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "hf"
_REQUIRED = [
    "bkai-foundation-models__vietnamese-bi-encoder",
    "BAAI__bge-reranker-v2-m3",
]
if all((_MODELS_DIR / r).is_dir() and any((_MODELS_DIR / r).iterdir()) for r in _REQUIRED):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
