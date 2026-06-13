"""Pipeline orchestrator.

  python -m vietjet.ingestion.pipeline download    — snapshot HF models → models/hf/ (offline cache)
  python -m vietjet.ingestion.pipeline crawl       — Firecrawl URLs → raw_data/
  python -m vietjet.ingestion.pipeline clean       — raw_data → clean_data
  python -m vietjet.ingestion.pipeline normalize   — clean_data → normalized_data (optional, không dùng để insert DB)
  python -m vietjet.ingestion.pipeline chunk       — clean_data → chunks.jsonl
  python -m vietjet.ingestion.pipeline ingest      — chunks.jsonl → pgvector + bm25.pkl
  python -m vietjet.ingestion.pipeline all         — download → crawl → clean → chunk → ingest
  python -m vietjet.ingestion.pipeline rebuild     — clean → chunk → ingest (skip crawl + download + normalize)
"""

from __future__ import annotations
import asyncio
import sys

from vietjet.ingestion import chunk as chunk_mod
from vietjet.ingestion import clean as clean_mod
from vietjet.ingestion import crawl as crawl_mod
from vietjet.ingestion import download_models as download_mod
from vietjet.ingestion import ingest as ingest_mod
from vietjet.ingestion import normalize as normalize_mod

STAGES = {
    "download": download_mod.main,
    "crawl": lambda: asyncio.run(crawl_mod.crawl_all()),
    "clean": clean_mod.clean_folder,
    "normalize": normalize_mod.normalize_folder,
    "chunk": chunk_mod.chunk_folder,
    "ingest": ingest_mod.main,
}

GROUPS = {
    "all": ["download", "crawl", "clean", "chunk", "ingest"],
    "rebuild": ["clean", "chunk", "ingest"],
}


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    stages = GROUPS.get(cmd, [cmd])
    for s in stages:
        if s not in STAGES:
            print(f"Unknown stage: {s}")
            sys.exit(1)
        print(f"\n=== Stage: {s} ===")
        STAGES[s]()


if __name__ == "__main__":
    main()
