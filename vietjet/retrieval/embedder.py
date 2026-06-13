from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from vietjet.config import EMBED_MODEL, RETRIEVAL_DEVICE


def pick_device() -> str:
    if RETRIEVAL_DEVICE and RETRIEVAL_DEVICE != "auto":
        return RETRIEVAL_DEVICE
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


_embedder: HuggingFaceEmbeddings | None = None


def get_embedder() -> HuggingFaceEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": pick_device()},
        )
    return _embedder
