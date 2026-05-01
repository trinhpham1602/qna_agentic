from typing import List, Union
import re
from dataclasses import dataclass, field
from rapidfuzz import fuzz
import json
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verbs that signal the user wants to perform an action (booking mode)
BOOKING_VERBS: set[str] = {
    "dat", "huy", "doi", "hoan", "bao_luu", "mua_them",
    "chon", "dang_ky", "thanh_toan", "check_in", "bao_cao",
}
# Verbs that signal the user wants information (Q&A mode)
QNA_VERBS: set[str] = {"hoi", "kiem_tra", "tim_kiem", "xac_nhan"}

# Regex patterns for booking-mode detection (fallback when no intents matched)
_BOOKING_RE = re.compile(
    r"\b(đặt|book|mua vé|mua thêm|thêm hành lý|hủy|cancel|đổi vé|đổi ngày|"
    r"hoàn vé|hoàn tiền|refund|bảo lưu|chọn ghế|seat selection|thanh toán|"
    r"check.in|đăng ký dịch vụ|báo cáo|khiếu nại)\b",
    re.IGNORECASE,
)
_QNA_RE = re.compile(
    r"\b(hỏi|cho hỏi|muốn biết|thắc mắc|chính sách|quy định|điều khoản|"
    r"phí là bao nhiêu|giá vé|hành lý được|được không|có được không|"
    r"giấy tờ|cần gì|như thế nào|tìm hiểu|tra cứu)\b",
    re.IGNORECASE,
)


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


def detect_intent_mode(
    text: str,
    matched_intents: list[dict] | None = None,
) -> dict:
    """
    Detect whether the user wants to perform a booking action or just get information.

    Primary signal: verb canonical of matched intents (most reliable).
    Fallback: regex pattern scoring on raw text.

    Returns:
        {
            "mode": "booking" | "qna" | "ambiguous",
            "confidence": float (0.0–1.0),
            "booking_score": int,
            "qna_score": int,
        }
    """
    booking_score = 0
    qna_score = 0

    # Primary: count booking vs qna verbs in matched intents
    if matched_intents:
        for intent in matched_intents:
            canon = intent.get("canonical", "")
            if canon in BOOKING_VERBS:
                booking_score += 2
            elif canon in QNA_VERBS:
                qna_score += 2

    # Fallback: regex scoring on raw text
    if booking_score == 0 and qna_score == 0:
        booking_score = len(_BOOKING_RE.findall(text))
        qna_score = len(_QNA_RE.findall(text))

    total = booking_score + qna_score
    if total == 0:
        return {"mode": "ambiguous", "confidence": 0.5, "booking_score": 0, "qna_score": 0}

    booking_ratio = booking_score / total
    if booking_ratio >= 0.65:
        return {"mode": "booking", "confidence": round(booking_ratio, 3), "booking_score": booking_score, "qna_score": qna_score}
    if booking_ratio <= 0.35:
        return {"mode": "qna", "confidence": round(1 - booking_ratio, 3), "booking_score": booking_score, "qna_score": qna_score}
    return {"mode": "ambiguous", "confidence": round(max(booking_ratio, 1 - booking_ratio), 3), "booking_score": booking_score, "qna_score": qna_score}


_CLAUSE_SPLIT_RE = re.compile(
    r"\s*\b(?:rồi|sau\s+đó|xong\s+rồi|xong|đồng\s+thời|vs\.?|"
    r"sau\s+khi|và\s+(?=[a-záàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ])|"
    r"cũng\s+(?=[a-z])|thêm\s+(?=[a-záàảãạăắằẳẵặâấầẩẫậđ]))\b\s*",
    re.IGNORECASE | re.UNICODE,
)


def split_into_clauses(text: str) -> list[str]:
    """
    Split a Vietnamese compound query into individual intent clauses.

    Splits on: rồi, sau đó, xong, vs, đồng thời, và <verb>, cũng <verb>, thêm <verb>
    Short fragments (< 2 tokens) are merged back to avoid noise.
    """
    raw = _CLAUSE_SPLIT_RE.split(text)
    clauses: list[str] = []
    pending = ""
    for part in raw:
        part = part.strip()
        if not part:
            continue
        combined = (pending + " " + part).strip() if pending else part
        if len(combined.split()) >= 2:
            clauses.append(combined)
            pending = ""
        else:
            pending = combined
    if pending:
        if clauses:
            clauses[-1] = (clauses[-1] + " " + pending).strip()
        else:
            clauses.append(pending)
    return clauses if clauses else [text]


def extract_multi_intent(
    text: str,
    intent_dataset: List["Entity"],
    entity_dataset: List["Entity"],
    threshold: int = 50,
) -> list[dict]:
    """
    Extract multiple (clause, intents, entities) tuples for compound queries.

    For single-clause queries returns a list with one element.
    Each element:
        {
            "clause":   str,          # the clause text
            "intents":  list[dict],   # matched verb intents for this clause
            "entities": list[dict],   # matched noun entities for this clause
        }
    """
    clauses = split_into_clauses(text)

    results: list[dict] = []
    for clause in clauses:
        intents  = extract_entities(clause, intent_dataset,  threshold)
        entities = extract_entities(clause, entity_dataset,  threshold)
        if intents or entities:
            results.append({"clause": clause, "intents": intents, "entities": entities})

    # Fallback: whole text if no clause produced results
    if not results:
        results.append({
            "clause":   text,
            "intents":  extract_entities(text, intent_dataset,  threshold),
            "entities": extract_entities(text, entity_dataset, threshold),
        })

    return results


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
