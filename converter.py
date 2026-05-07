from utils import get_intent_classifier

def test_claude_dataset_v1():
    classifier = get_intent_classifier()
    proba = classifier.predict_proba(["doi ve co mat phi khong"])

    print(classifier.classes_)
    print(proba)


if __name__ == "__main__":
    test_claude_dataset_v1()

