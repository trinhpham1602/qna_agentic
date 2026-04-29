from typing import List, Union
import re
from dataclasses import dataclass, field
from rapidfuzz import fuzz
import json
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Entity:
    id: Union[str, int]
    canonical: str
    anchors: List[str] = field(default_factory=list)
    type: str = "entity"
    risk: int = 1
    label: str = ""


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_entities(text: str, anchor_dataset: List[Entity], threshold: int = 60) -> List[dict]:
    """
    Anchor-first sliding window matching.

    For each anchor, slides a window of the same token length across the
    normalized text. Returns the best match per canonical with its metadata
    (type, risk, label) included so callers don't need a second lookup.

    Complexity: O(entities × anchors × n)
    """
    norm_text = normalize(text)
    tokens = norm_text.split()
    n = len(tokens)

    best: dict[str, dict] = {}

    for entity in anchor_dataset:
        for anchor in entity.anchors:
            norm_anchor = normalize(anchor)
            anchor_tokens = norm_anchor.split()
            window = len(anchor_tokens)
            if window == 0 or window > n:
                continue

            for i in range(n - window + 1):
                phrase = " ".join(tokens[i : i + window])
                score = fuzz.token_sort_ratio(phrase, norm_anchor)

                prev = best.get(entity.canonical)
                if prev is None or score > prev["best_score"]:
                    best[entity.canonical] = {
                        "canonical": entity.canonical,
                        "best_score": score,
                        "matched_anchor": anchor,
                        "matched_span": phrase,
                        "type": entity.type,
                        "risk": entity.risk,
                        "label": entity.label,
                    }

    return [v for v in best.values() if v["best_score"] >= threshold]


if __name__ == "__main__":
    with open("dataset/entities.json") as f:
        raw = json.load(f)
    dataset = [Entity(**item) for item in raw]

    with open("dataset/intents.json") as f:
        raw_intents = json.load(f)
    intent_dataset = [Entity(**item) for item in raw_intents]

    q = "vj co hoan ve được khong?"
    entities = extract_entities(q, dataset, threshold=55)
    intents = extract_entities(q, intent_dataset, threshold=55)
    logger.info("Entities: %s", entities)
    logger.info("Intents:  %s", intents)
