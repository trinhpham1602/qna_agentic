from typing import TypedDict
from typing import List
import re
from dataclasses import dataclass, field
from typing import List
from rapidfuzz import fuzz
import json


@dataclass
class Entity(TypedDict):
    id: int
    canonical: str
    anchors: List[str] = field(default_factory=list)

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)     # remove extra spaces
    return text.strip()

def extract_entities(text: str, anchor_dataset: List[Entity]):
    tokens = text.split()
    n = len(tokens)
    results = []

    for i in range(n):
        for j in range(i + 1, n + 1):
            phrase = " ".join(tokens[i:j])
            best_score = 0
            canonical = ""
            for item in anchor_dataset:
                item = Entity(**item)
                max_score = max([fuzz.ratio(phrase, e) for e in item.get("canonical")])
                if max_score >= best_score:
                    best_score = max_score
                    canonical = item.get("canonical")
            exists =[e for e in results if e.get("canonical") == canonical]
            if len(exists) == 0:
                results.append({
                    "best_score": best_score,
                    "canonical": canonical,
                })
            elif exists[0].get("best_score") < best_score:
                for e in results:
                    if e.get("canonical") == canonical:
                        e["best_score"] = best_score
    return results
with open("./entity/anchors_dataset.json", "r") as f:
    data = json.load(f)
    all = extract_entities("vietjet co hoan ve khong?", [Entity(**item) for item in data])
    top_k =  [e for e in all if e.get("best_score") >= 50]
    print(top_k)
