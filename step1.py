from rapidfuzz import fuzz
query = "vj có được hoàn vé không?"
texts = [
      "dịch vụ đặc biệt",
      "special service",
      "ssr",
      "yêu cầu đặc biệt",
      "dịch vụ bổ sung",
      "hành khách đặt biệt"
    ]
score = [fuzz.ratio("hành khách đặt biệt", e) for e in texts]
print(score)