if __name__ == "__main__":
    import json
    from utils import MultiLabelExtractor
    entity_ext = MultiLabelExtractor()

    with open("dataset/entities.json") as f:
        entity_data = json.load(f)
        entity_ext.fit(entity_data, is_intent=False)

    with open("faq_data.json", "r") as f:
        data = json.load(f)
        result = {}
        for e in data:
            item = {}
            ents = entity_ext.predict_single(e.get("question", ""), threshold=0.3)
            if len(ents) > 0:
                first = ents[0]
                if result.get(first.get("canonical")) is None:
                    item["canonical"] = first.get("canonical")
                    item["type"] = first.get("type")
                    item["questions"] = [e.get("question")]
                    result[first.get("canonical")] = item
                else:
                    result[first.get("canonical")]["questions"].append(e.get("question"))

    with open("test.json", "w") as f:
        json.dump(list(result.values()), f, ensure_ascii=False, indent=2)     
