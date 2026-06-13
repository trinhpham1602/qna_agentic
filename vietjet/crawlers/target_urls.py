from __future__ import annotations

from vietjet.config import DOC_TYPE_MAP, URLS


def filter_target_urls(doc_type: str | None = None) -> list[str]:
    if doc_type is None or doc_type == "other":
        return [u["url"] for u in URLS]
    matched = [
        u["url"]
        for u in URLS
        if DOC_TYPE_MAP.get(u["filename"]) == doc_type
    ]
    return matched or [u["url"] for u in URLS]


def all_target_urls() -> list[str]:
    return [u["url"] for u in URLS]
