from __future__ import annotations

import re
import unicodedata
from typing import Optional

from vietjet.cache.store import CacheStore, stable_hash
from vietjet.config import TTL_NORMALIZE


_AIRLINE_SYNONYMS = {
    "vj": "vietjet",
    "vietjet": "vietjet",
    "vietjetair": "vietjet",
    "vja": "vietjet",
}

_CLASS_SYNONYMS = {
    "eco": "eco",
    "deluxe": "deluxe",
    "skyboss": "skyboss",
    "business": "business",
    "thuong gia": "business",
}

_ROUTE_SYNONYMS = {
    "noi dia": "dom",
    "trong nuoc": "dom",
    "quoc te": "intl",
    "nuoc ngoai": "intl",
}

_POLICY_SYNONYMS = {
    "doi ve": "ticket_change",
    "hoan ve": "ticket_refund",
    "bao luu": "ticket_reserve",
    "hanh ly": "baggage",
    "thanh toan": "payment",
    "boi thuong": "compensation",
    "phi": "fee",
    "le phi": "fee",
}


_PUNCT_RE = re.compile(r"[^\w\s]")
_MULTI_WS = re.compile(r"\s+")


_VIETNAMESE_D_MAP = str.maketrans({"đ": "d", "Đ": "D"})


def strip_diacritics(text: str) -> str:
    text = text.translate(_VIETNAMESE_D_MAP)
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def normalize_query(raw: str) -> dict:
    raw = raw.strip()
    if not raw:
        return {"raw_question": "", "normalized_query": "", "slots": {}}

    flat = strip_diacritics(raw.lower())
    flat = _PUNCT_RE.sub(" ", flat)
    flat = _MULTI_WS.sub(" ", flat).strip()

    slots: dict[str, str] = {}
    tokens = flat.split()

    for token in tokens:
        if token in _AIRLINE_SYNONYMS:
            slots["airline"] = _AIRLINE_SYNONYMS[token]
        if token in _CLASS_SYNONYMS:
            slots["ticket_class"] = _CLASS_SYNONYMS[token]

    for phrase, slot_value in _ROUTE_SYNONYMS.items():
        if phrase in flat:
            slots["route_type"] = slot_value
            break

    for phrase, slot_value in _POLICY_SYNONYMS.items():
        if phrase in flat:
            slots.setdefault("group_policy", slot_value)

    canonical_parts: list[str] = []
    if slots.get("airline"):
        canonical_parts.append(slots["airline"])
    if slots.get("ticket_class"):
        canonical_parts.append(slots["ticket_class"])
    if slots.get("route_type"):
        canonical_parts.append(slots["route_type"])
    if slots.get("group_policy"):
        canonical_parts.append(slots["group_policy"])

    if canonical_parts:
        normalized = " ".join(canonical_parts) + " | " + flat
    else:
        normalized = flat

    return {
        "raw_question": raw,
        "normalized_query": normalized,
        "slots": slots,
    }


async def normalize_query_with_cache(
    cache: CacheStore, raw: str
) -> dict:
    h = stable_hash(raw)
    key = cache.build_key("norm", "v1", h)
    cached = await cache.get_json(key)
    if cached:
        return cached
    result = normalize_query(raw)
    await cache.set_json(key, result, ttl_seconds=TTL_NORMALIZE)
    return result
