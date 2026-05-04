if __name__ == "__main__":
    import json
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    # ===== LOAD DATA =====
    def load_dataset(path, label):
        with open(path, "r") as f:
            data = json.load(f)
        X = [e.get("text") for e in data]
        y = [label] * len(X)
        return X, y

    CMD_DS, CMD_DS_L = load_dataset("dataset/flight_commands_20k.json", "command")
    Q_DS, Q_DS_L = load_dataset("dataset/flight_questions_20k.json", "question")
    N_DS, N_DS_L = load_dataset("dataset/random_noise_20k.json", "noise")

    # ===== MERGE =====
    X_all = CMD_DS + Q_DS + N_DS
    y_all = CMD_DS_L + Q_DS_L + N_DS_L

    # ===== MODEL =====
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=50000
    )

    X_train = vectorizer.fit_transform(X_all)

    classifier = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs"
    )

    classifier.fit(X_train, y_all)

    # ===== TEST =====
    X = vectorizer.transform(["lieu rang booking sớm có re hon"])
    proba = classifier.predict_proba(X)

    print(classifier.classes_)
    print(proba)

