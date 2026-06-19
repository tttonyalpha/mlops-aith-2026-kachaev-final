from __future__ import annotations

import argparse
import json

import joblib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local prediction helper for smoke checks only.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--text", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model_path)
    label = model.predict([args.text])[0]
    result = {"label": str(label), "text": args.text}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([args.text])[0]
        result["probabilities"] = {
            str(label_name): float(value)
            for label_name, value in zip(model.classes_, probabilities)
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
