from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


LABELS = ["negative", "positive"]


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    required = {"text", "label"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    data = data.dropna(subset=["text", "label"]).copy()
    data["text"] = data["text"].astype(str)
    data["label"] = data["label"].astype(str).str.lower()
    unknown = sorted(set(data["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"Unexpected labels: {unknown}; expected {LABELS}")
    return data


def build_pipeline(
    classifier: str = "logreg",
    c: float = 1.0,
    ngram_max: int = 1,
    max_features: int = 5000,
) -> Pipeline:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, ngram_max),
        max_features=max_features,
    )
    if classifier == "logreg":
        estimator = LogisticRegression(C=c, max_iter=1000, random_state=42)
    elif classifier == "linearsvc":
        estimator = LinearSVC(C=c, random_state=42)
    else:
        raise ValueError("classifier must be one of: logreg, linearsvc")
    return Pipeline([("tfidf", vectorizer), ("classifier", estimator)])

