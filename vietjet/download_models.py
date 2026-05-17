"""Download required HF models to models/hf/ for fully-offline loading."""

from __future__ import annotations

from vietjet.config import EMBED_MODEL_ID, RERANK_MODEL_ID
from vietjet.models import download


def main() -> None:
    for repo in (EMBED_MODEL_ID, RERANK_MODEL_ID):
        download(repo)


if __name__ == "__main__":
    main()
