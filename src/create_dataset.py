from __future__ import annotations

import argparse
from pathlib import Path

from clearml import Dataset, Task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a versioned ClearML Dataset.")
    parser.add_argument("--project", default="mlops-aith-2026-kachaev")
    parser.add_argument("--dataset-name", default="sentiment_texts")
    parser.add_argument("--dataset-version", default="1.0.0")
    parser.add_argument("--data-path", default="data/raw/sentiment.csv")
    parser.add_argument("--output-id-file", default="outputs/dataset_id.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    task = Task.init(
        project_name=args.project,
        task_name=f"create dataset {args.dataset_name} v{args.dataset_version}",
        task_type=Task.TaskTypes.data_processing,
    )
    task.connect(vars(args), name="dataset_params")

    dataset = Dataset.create(
        dataset_project=args.project,
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
    )
    dataset.add_files(path=str(data_path), dataset_path="raw")
    dataset.upload()
    dataset.finalize()

    output_file = Path(args.output_id_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(dataset.id + "\n", encoding="utf-8")

    task.upload_artifact("dataset_id", artifact_object=dataset.id)
    task.close()
    print(f"ClearML dataset_id: {dataset.id}")
    print(f"Saved dataset id to: {output_file}")


if __name__ == "__main__":
    main()

