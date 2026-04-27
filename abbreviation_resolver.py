# abbreviation_resolver.py

from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from rapidfuzz import fuzz


@dataclass
class AbbreviationEntry:
    key: str
    variants: List[str]
    expansions: List[str]


class AbbreviationResolver:
    def __init__(
        self,
        entries: List[AbbreviationEntry],
        fuzzy_threshold: int = 85,
        max_expansions_per_abbr: int = 2,
    ):
        self.entries = entries
        self.fuzzy_threshold = fuzzy_threshold
        self.max_expansions_per_abbr = max_expansions_per_abbr

    # ---------------------------
    # 1. Detect abbreviation
    # ---------------------------
    def _match_token(self, token: str) -> Optional[AbbreviationEntry]:
        token = token.lower()

        # 1. exact / variant match
        for entry in self.entries:
            if token == entry.key or token in entry.variants:
                return entry

        # 2. fuzzy match
        best_entry = None
        best_score = 0

        for entry in self.entries:
            for v in [entry.key] + entry.variants:
                score = fuzz.ratio(token, v)
                if score > best_score:
                    best_score = score
                    best_entry = entry

        if best_score >= self.fuzzy_threshold:
            return best_entry

        return None

    # ---------------------------
    # 2. Extract abbreviations from query
    # ---------------------------
    def detect(self, query: str) -> Dict[str, AbbreviationEntry]:
        tokens = query.lower().split()
        found = {}

        for t in tokens:
            match = self._match_token(t)
            if match:
                found[t] = match

        return found

    # ---------------------------
    # 3. Expand query (multi-query)
    # ---------------------------
    def expand_query(self, query: str) -> List[str]:
        found = self.detect(query)

        # no abbreviation → return original
        if not found:
            return [query]

        queries = [query]

        for token, entry in found.items():
            new_queries = []

            expansions = entry.expansions[: self.max_expansions_per_abbr]

            for q in queries:
                for exp in expansions:
                    new_q = q.replace(token, exp)
                    new_queries.append(new_q)

            queries.extend(new_queries)

        # remove duplicates
        queries = list(set(queries))

        return queries

entries = [
    AbbreviationEntry(
        key="ecom",
        variants=["eco", "e-commerce", "ecommerce", "e-com"],
        expansions=["e-commerce", "online retail"],
    ),
    AbbreviationEntry(
        key="ads",
        variants=["ad", "advert"],
        expansions=["advertising", "paid media"],
    ),
]

resolver = AbbreviationResolver(entries)

query = "e-com có hành lý xách tay 7kg dang ads không"
query = " ".join([e.lower() for e in query.split()])
expanded = resolver.expand_query(query)

for q in expanded:
    print(q)

# mapping tất cả các case có thể

# cái mình muốn là lấy ra câu hợp lý nhất.