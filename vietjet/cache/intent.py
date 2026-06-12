from __future__ import annotations

import re


_REALTIME_PATTERNS = [
    re.compile(r"\bm[ơo]i nh[aâ]t\b", re.I),
    re.compile(r"\bh[ơo]m nay\b", re.I),
    re.compile(r"\bhi[ệe]n t[aạ]i\b", re.I),
    re.compile(r"\bb[aâ]y gi[ơo]\b", re.I),
    re.compile(r"\bgi[aá] v[eé]\b", re.I),
    re.compile(r"\bgi[aá] r[eẻ]\b", re.I),
    re.compile(r"\bstatus\b", re.I),
    re.compile(r"\bbooking\b", re.I),
    re.compile(r"\btr[aạ]ng th[aá]i\b", re.I),
    re.compile(r"\bchuy[eế]n bay\b.*\b(hôm nay|ng[aà]y mai)\b", re.I),
    re.compile(r"\bkhuy[eế]n m[aã]i\b", re.I),
    re.compile(r"\bs[aắ]p t[ơo]i\b", re.I),
    re.compile(r"\blatest\b", re.I),
    re.compile(r"\bcurrent\b", re.I),
    re.compile(r"\bnow\b", re.I),
    re.compile(r"\btoday\b", re.I),
]


def is_realtime_intent(query: str) -> bool:
    if not query:
        return False
    return any(p.search(query) for p in _REALTIME_PATTERNS)
