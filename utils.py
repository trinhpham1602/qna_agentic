from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union
import re

from graph_db import KnowledgeGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOOKING_VERBS: set[str] = {
    "dat", "huy", "doi", "hoan", "bao_luu", "mua_them",
    "chon", "dang_ky", "thanh_toan", "check_in", "bao_cao",
}
QNA_VERBS: set[str] = {"hoi", "kiem_tra", "tim_kiem", "xac_nhan"}

_ENTITY_TEMPLATES: list[str] = [
    "{anchor}",
    "hỏi về {anchor}",
    "thông tin về {anchor}",
    "quy định {anchor}",
    "chính sách {anchor}",
    "tôi muốn biết về {anchor}",
    "{anchor} như thế nào",
    "cho hỏi {anchor}",
    "{anchor} là gì",
    "{anchor} được không",
    "tôi có {anchor} không",
    "kiểm tra {anchor}",
    "{anchor} của VietJet",
    "VietJet {anchor}",
    "liên quan đến {anchor}",
    "{anchor} bao nhiêu",
    "phí {anchor}",
    "{anchor} tính thế nào",
    "{anchor} có phí không",
    "cần biết {anchor}",
]

_INTENT_TEMPLATES: list[str] = [
    "{anchor}",
    "tôi muốn {anchor}",
    "làm sao để {anchor}",
    "cần {anchor}",
    "mình muốn {anchor}",
    "{anchor} được không",
    "có thể {anchor} không",
    "tôi cần {anchor}",
    "muốn {anchor}",
    "mình cần {anchor}",
    "tôi {anchor}",
    "hướng dẫn {anchor}",
    "{anchor} như thế nào",
    "cách {anchor}",
    "thủ tục {anchor}",
    "{anchor} ở đâu",
    "{anchor} khi nào",
    "{anchor} bao nhiêu",
    "cho tôi {anchor}",
    "mình {anchor} được không",
]


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


class MultiLabelExtractor:
    def __init__(self) -> None:
        self.vectorizer = None
        self.classifier = None
        self.binarizer = None
        self.label_meta: dict[str, dict] = {}
        self._fitted: bool = False

    def _build_samples(
        self,
        dataset: list[dict],
        is_intent: bool,
    ) -> tuple[list[str], list[list[str]]]:
        templates = _INTENT_TEMPLATES if is_intent else _ENTITY_TEMPLATES
        texts: list[str] = []
        label_sets: list[list[str]] = []

        per_item_samples: dict[str, list[str]] = {}

        for item in dataset:
            item_id = str(item["id"])
            self.label_meta[item_id] = {
                "type": item.get("type", "entity"),
                "risk": item.get("risk", 1),
                "label": item.get("label", item.get("canonical", "")),
            }
            anchors = item.get("anchors", [item.get("canonical", "")])
            item_samples: list[str] = []
            for anchor in anchors:
                for tmpl in templates:
                    sample = tmpl.format(anchor=anchor)
                    texts.append(sample)
                    label_sets.append([item_id])
                    item_samples.append(sample)
            per_item_samples[item_id] = item_samples

        all_ids = list(per_item_samples.keys())
        if len(all_ids) >= 2:
            for _ in range(300):
                id_a, id_b = random.sample(all_ids, 2)
                sample_a = random.choice(per_item_samples[id_a])
                sample_b = random.choice(per_item_samples[id_b])
                combined = sample_a + " và " + sample_b
                texts.append(combined)
                label_sets.append([id_a, id_b])

        return texts, label_sets

    def fit(self, dataset: list[dict], is_intent: bool = False) -> "MultiLabelExtractor":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.preprocessing import MultiLabelBinarizer

        texts, label_sets = self._build_samples(dataset, is_intent)

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 7), max_features=50000
        )
        self.binarizer = MultiLabelBinarizer()
        self.classifier = OneVsRestClassifier(
            LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        )

        X = self.vectorizer.fit_transform(texts)
        Y = self.binarizer.fit_transform(label_sets)
        self.classifier.fit(X, Y)
        self._fitted = True

        logger.info(
            "MultiLabelExtractor fitted: %d samples, %d classes",
            len(texts),
            len(self.binarizer.classes_),
        )
        return self

    def train_evaluate(
        self,
        dataset: list[dict],
        is_intent: bool = False,
        test_size: float = 0.2,
        seed: int = 42,
    ) -> dict:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.preprocessing import MultiLabelBinarizer

        texts, label_sets = self._build_samples(dataset, is_intent)

        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, label_sets, test_size=test_size, random_state=seed
        )

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), max_features=50000
        )
        self.binarizer = MultiLabelBinarizer()
        self.classifier = OneVsRestClassifier(
            LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        )

        X_train = self.vectorizer.fit_transform(train_texts)
        Y_train = self.binarizer.fit_transform(train_labels)
        self.classifier.fit(X_train, Y_train)
        self._fitted = True

        metrics = self.evaluate(test_texts, test_labels)
        metrics["train_size"] = len(train_texts)
        metrics["test_size"] = len(test_texts)
        return metrics

    def predict_single(self, text: str, threshold: float = 0.3) -> list[dict]:
        if not self._fitted:
            raise RuntimeError("MultiLabelExtractor is not fitted yet.")

        X = self.vectorizer.transform([text])
        proba = self.classifier.predict_proba(X)[0]
        classes = self.binarizer.classes_

        results = []
        for cls, prob in zip(classes, proba):
            if prob >= threshold:
                meta = self.label_meta.get(cls, {})
                results.append(
                    {
                        "canonical": cls,
                        "confidence": float(prob),
                        "type": meta.get("type", "entity"),
                        "risk": meta.get("risk", 1),
                        "label": meta.get("label", cls),
                        "best_score": float(prob),
                        "matched_span": text,
                    }
                )

        if not results:
            best_idx = int(proba.argmax())
            cls = classes[best_idx]
            prob = float(proba[best_idx])
            meta = self.label_meta.get(cls, {})
            results.append(
                {
                    "canonical": cls,
                    "confidence": prob,
                    "type": meta.get("type", "entity"),
                    "risk": meta.get("risk", 1),
                    "label": meta.get("label", cls),
                    "best_score": prob,
                    "matched_span": text,
                }
            )

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    def evaluate(
        self,
        test_texts: list[str],
        test_labels: list[list[str]],
        threshold: float = 0.3,
    ) -> dict:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            hamming_loss,
        )

        X_test = self.vectorizer.transform(test_texts)
        proba = self.classifier.predict_proba(X_test)
        Y_pred = (proba >= threshold).astype(int)
        Y_true = self.binarizer.transform(test_labels)

        report = classification_report(
            Y_true, Y_pred, output_dict=True, zero_division=0
        )
        exact = float(accuracy_score(Y_true, Y_pred))
        hl = float(hamming_loss(Y_true, Y_pred))
        micro_f1 = report.get("micro avg", {}).get("f1-score", 0.0)
        macro_f1 = report.get("macro avg", {}).get("f1-score", 0.0)

        per_class = {}
        for cls_idx, cls_name in enumerate(self.binarizer.classes_):
            cls_str = str(cls_idx)
            if cls_str in report:
                per_class[cls_name] = report[cls_str]

        return {
            "exact_match": exact,
            "hamming_loss": hl,
            "micro_f1": float(micro_f1),
            "macro_f1": float(macro_f1),
            "per_class": per_class,
        }

    def save(self, path: str) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "classifier": self.classifier,
                "binarizer": self.binarizer,
                "label_meta": self.label_meta,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "MultiLabelExtractor":
        import joblib

        data = joblib.load(path)
        obj = cls()
        obj.vectorizer = data["vectorizer"]
        obj.classifier = data["classifier"]
        obj.binarizer = data["binarizer"]
        obj.label_meta = data["label_meta"]
        obj._fitted = True
        return obj


_CLAUSE_SPLIT_RE = re.compile(
    r"\s*\b(?:rồi|sau\s+đó|xong\s+rồi|xong|đồng\s+thời|vs\.?|"
    r"sau\s+khi|và\s+(?=[a-záàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ])|"
    r"cũng\s+(?=[a-z])|thêm\s+(?=[a-záàảãạăắằẳẵặâấầẩẫậđ]))\b\s*",
    re.IGNORECASE | re.UNICODE,
)


def split_into_clauses(text: str) -> list[str]:
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


def detect_intent_mode(text: str, matched_intents: list[dict] | None = None) -> dict:
    booking_score = 0.0
    qna_score = 0.0
    if matched_intents:
        for i in matched_intents:
            w = i.get("confidence", i.get("best_score", 50) / 100)
            if i.get("canonical", "") in BOOKING_VERBS:
                booking_score += w
            elif i.get("canonical", "") in QNA_VERBS:
                qna_score += w
    total = booking_score + qna_score
    if total == 0:
        return {"mode": "ambiguous", "confidence": 0.5, "booking_score": 0.0, "qna_score": 0.0}
    ratio = booking_score / total
    if ratio >= 0.6:
        return {"mode": "booking", "confidence": round(ratio, 3), "booking_score": booking_score, "qna_score": qna_score}
    if ratio <= 0.4:
        return {"mode": "qna", "confidence": round(1 - ratio, 3), "booking_score": booking_score, "qna_score": qna_score}
    return {"mode": "ambiguous", "confidence": round(max(ratio, 1 - ratio), 3), "booking_score": booking_score, "qna_score": qna_score}


def extract_multi_intent(
    text: str,
    entity_extractor: "MultiLabelExtractor",
    intent_extractor: "MultiLabelExtractor",
    threshold: float = 0.3,
) -> list[dict]:
    clauses = split_into_clauses(text)
    results = []
    for clause in clauses:
        intents = intent_extractor.predict_single(clause, threshold)
        entities = entity_extractor.predict_single(clause, threshold)
        if intents or entities:
            results.append({"clause": clause, "intents": intents, "entities": entities})
    if not results:
        results.append({
            "clause": text,
            "intents": intent_extractor.predict_single(text, threshold),
            "entities": entity_extractor.predict_single(text, threshold),
        })
    return results


def find_paths(
    seed_text: str,
    entity_extractor: "MultiLabelExtractor",
    intent_extractor: "MultiLabelExtractor",
    kg: "KnowledgeGraph",
    entity_labels: dict[str, str] | None = None,
    intent_labels: dict[str, str] | None = None,
    depth: int = 2,
    threshold: float = 0.3,
    relation_filter: set[str] | None = None,
) -> dict:
    """
    Extract entities/intents from text, then traverse the KG from each matched
    entity/intent node.  Returns a summary dict with matched labels and found paths.
    """
    entity_labels = entity_labels or {}
    intent_labels = intent_labels or {}

    matched_entities = entity_extractor.predict_single(seed_text, threshold)
    matched_intents  = intent_extractor.predict_single(seed_text, threshold)

    seed_nodes: set[str] = set()
    for e in matched_entities:
        seed_nodes.add(e["canonical"])
    for i in matched_intents:
        seed_nodes.add(i["canonical"])

    triplets = kg.traverse(seed_nodes, depth=depth, relation_filter=relation_filter)

    def _label(node_id: str) -> str:
        return entity_labels.get(node_id) or intent_labels.get(node_id) or node_id

    paths = [
        {
            "from": t[0], "from_label": _label(t[0]),
            "relation": t[1], "relation_label": intent_labels.get(t[1], t[1]),
            "to": t[2], "to_label": _label(t[2]),
        }
        for t in triplets
    ]

    return {
        "seed_text": seed_text,
        "entities": [(e["canonical"], round(e["confidence"], 2)) for e in matched_entities],
        "intents":  [(i["canonical"], round(i["confidence"], 2)) for i in matched_intents],
        "seed_nodes": sorted(seed_nodes),
        "paths": paths,
    }


if __name__ == "__main__":
    import json

    with open("dataset/entities.json") as f:
        entity_data = json.load(f)
    with open("dataset/intents.json") as f:
        intent_data = json.load(f)

    print("=== Training Extractors ===")
    entity_ext = MultiLabelExtractor()
    entity_ext.fit(entity_data, is_intent=False)
    intent_ext = MultiLabelExtractor()
    intent_ext.fit(intent_data, is_intent=True)

    kg = KnowledgeGraph.from_dataset("dataset/claude_dataset")

    entity_labels = {e["id"]: e.get("label", e["id"]) for e in entity_data}
    intent_labels = {i["id"]: i.get("label", i["id"]) for i in intent_data}

    tests = [
        "quy định phụ nữ mang thai và hành lý xách tay",
        "phí đổi vé skyboss nội địa",
        "trẻ đi một mình cần giấy tờ gì",
        "mua thêm ký gửi 20kg vietjet thì cần làm gì",
        "nâng hạng lên deluxe mất bao nhiêu",
        "mã khuyến mãi nhập ở đâu",
    ]

    print("\n" + "=" * 60)
    print("=== Multi-Intent / Multi-Entity Extraction ===")
    print("=" * 60)
    for q in tests:
        ents = entity_ext.predict_single(q, threshold=0.3)
        ints = intent_ext.predict_single(q, threshold=0.3)
        mode = detect_intent_mode(q, ints)
        print(f"\nQ: {q}")
        print("  Intents : " + ", ".join(f"{i['canonical']}({i['confidence']:.2f})" for i in ints[:3]))
        print("  Entities: " + ", ".join(f"{e['canonical']}({e['confidence']:.2f})" for e in ents[:3]))
        print(f"  Mode    : {mode['mode']} ({mode['confidence']:.2f})")

    print("\n" + "=" * 60)
    print("=== KG Path Traversal ===")
    print("=" * 60)
    path_tests = [
        ("phụ nữ mang thai", None),
        ("đổi vé nội địa eco", None),
        ("pin dự phòng xách tay", None),
        ("nâng hạng skyboss", None),
    ]
    for text, rel_filter in path_tests:
        result = find_paths(
            text, entity_ext, intent_ext, kg,
            entity_labels=entity_labels, intent_labels=intent_labels,
            depth=2, relation_filter=rel_filter,
        )
        print(f"\nSeed: \"{result['seed_text']}\"")
        print(f"  Nodes : {result['seed_nodes']}")
        if result["paths"]:
            for p in result["paths"]:
                print(f"  {p['from_label']} --[{p['relation']}]--> {p['to_label']}")
        else:
            print("  (no paths found)")
