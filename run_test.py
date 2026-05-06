import random
import json

# ===== DEFINE LABEL SPACE =====
intents = {
    "ask_fee": [
        "phí {service} bao nhiêu",
        "{service} mất bao nhiêu tiền",
        "giá {service} là bao nhiêu",
        "{service} tốn phí không",
        "{service} có đắt không"
    ],
    "buy_service": [
        "tôi muốn mua {service}",
        "đặt {service} giúp tôi",
        "thêm {service}",
        "mua thêm {service}",
        "book {service}"
    ]
}

fee_types = {
    "baggage": ["hành lý", "hành lý ký gửi"],
    "change": ["đổi vé", "chuyển ngày bay"],
    "seat": ["chọn chỗ", "ghế ngồi"],
    "meal": ["suất ăn", "đồ ăn trên máy bay"]
}

scenarios = {
    "prepaid": ["mua trước", "đặt trước", "online"],
    "airport": ["tại sân bay"],
    "overweight": ["quá cân", "quá ký"],
    "domestic": ["nội địa"],
    "international": ["quốc tế"]
}

# ===== GENERATOR =====
def generate_sample():
    intent = random.choice(list(intents.keys()))
    fee_type = random.choice(list(fee_types.keys()))
    scenario = random.choice(list(scenarios.keys()))

    service = random.choice(fee_types[fee_type])
    scenario_text = random.choice(scenarios[scenario])

    template = random.choice(intents[intent])

    # 50% có scenario, 50% không (giống user thật)
    if random.random() > 0.5:
        text = template.format(service=f"{service} {scenario_text}")
    else:
        text = template.format(service=service)
        scenario = None  # missing info

    return {
        "text": text,
        "intent": intent,
        "fee_type": fee_type,
        "scenario": scenario
    }

# ===== GENERATE DATASET =====
dataset = [generate_sample() for _ in range(50000)]

# ===== SAVE =====
with open("flight_dataset_50k.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print("Done: generated 50k samples")