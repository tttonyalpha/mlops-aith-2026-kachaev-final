from __future__ import annotations

import sys

from src.train import main


sys.argv = [
    "train_logreg_agent.py",
    "--dataset-id",
    "caea8b717da64c0ba645cadecbf39ddd",
    "--experiment-name",
    "sentiment-logreg-agent",
    "--classifier",
    "logreg",
    "--c",
    "1.0",
    "--ngram-max",
    "1",
    "--max-features",
    "5000",
]

main()
