from __future__ import annotations

import sys

from src.train import main


sys.argv = [
    "train_linearsvc_agent.py",
    "--dataset-id",
    "caea8b717da64c0ba645cadecbf39ddd",
    "--experiment-name",
    "sentiment-linearsvc-agent",
    "--classifier",
    "linearsvc",
    "--c",
    "0.7",
    "--ngram-max",
    "2",
    "--max-features",
    "5000",
]

main()
