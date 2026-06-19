from src.modeling import build_pipeline, load_dataset

def test_pipeline_trains_on_sample_data():
    data = load_dataset("data/raw/sentiment.csv")
    model = build_pipeline(classifier="logreg", ngram_max=1)
    model.fit(data["text"], data["label"])
    prediction = model.predict(["The service was excellent and helpful"])[0]
    assert prediction in {"positive", "negative"}

