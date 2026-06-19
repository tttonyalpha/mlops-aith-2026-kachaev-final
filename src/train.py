from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
from PIL import Image, ImageDraw
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modeling import LABELS, build_pipeline, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ClearML-tracked text classifier.")
    parser.add_argument("--project", default="mlops-aith-2026-kachaev")
    parser.add_argument("--experiment-name", default="sentiment-logreg-baseline")
    parser.add_argument("--dataset-id", default=None, help="Required for real ClearML runs.")
    parser.add_argument("--local-data-path", default="data/raw/sentiment.csv")
    parser.add_argument("--queue", default="students")
    parser.add_argument("--execute-remotely", action="store_true")
    parser.add_argument("--local-only", action="store_true", help="Skip ClearML for quick local smoke tests.")
    parser.add_argument("--classifier", choices=["logreg", "linearsvc"], default="logreg")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--ngram-max", type=int, choices=[1, 2], default=1)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def resolve_dataset_path(args: argparse.Namespace, task) -> Path:
    if args.local_only:
        return Path(args.local_data_path)

    if not args.dataset_id:
        raise ValueError("--dataset-id is required unless --local-only is set")

    from clearml import Dataset

    dataset = Dataset.get(dataset_id=args.dataset_id)
    dataset_path = Path(dataset.get_local_copy())
    csv_files = sorted(dataset_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in ClearML dataset copy: {dataset_path}")
    task.upload_artifact("resolved_dataset_path", artifact_object=str(csv_files[0]))
    return csv_files[0]


def init_clearml(args: argparse.Namespace):
    if args.local_only:
        return None

    from clearml import Task

    if Path("requirements-agent.txt").exists() and hasattr(Task, "force_requirements_env_freeze"):
        Task.force_requirements_env_freeze(requirements_file="requirements-agent.txt")

    task = Task.init(
        project_name=args.project,
        task_name=args.experiment_name,
        task_type=Task.TaskTypes.training,
        auto_connect_frameworks=True,
    )
    params = vars(args).copy()
    task.connect(params, name="training_params")
    if args.execute_remotely:
        task.execute_remotely(queue_name=args.queue, exit_process=True)
    return task


def save_confusion_matrix_image(matrix, output_path: Path) -> None:
    image = Image.new("RGB", (520, 420), "white")
    draw = ImageDraw.Draw(image)
    draw.text((150, 20), "Validation confusion matrix", fill="black")
    draw.text((210, 65), "Predicted", fill="black")
    draw.text((20, 185), "Actual", fill="black")

    left, top, cell = 150, 100, 120
    headers = ["negative", "positive"]
    for idx, label in enumerate(headers):
        draw.text((left + idx * cell + 35, top - 28), label, fill="black")
        draw.text((left - 95, top + idx * cell + 50), label, fill="black")

    max_value = max(int(matrix.max()), 1)
    for row in range(2):
        for col in range(2):
            value = int(matrix[row, col])
            intensity = 235 - int(145 * value / max_value)
            color = (intensity, intensity + 10, 255)
            x0, y0 = left + col * cell, top + row * cell
            x1, y1 = x0 + cell, y0 + cell
            draw.rectangle((x0, y0, x1, y1), fill=color, outline="black", width=2)
            draw.text((x0 + 52, y0 + 50), str(value), fill="black")

    image.save(output_path)


def log_clearml_results(task, metrics: dict, model_path: Path, figure_path: Path) -> str | None:
    if task is None:
        return None

    logger = task.get_logger()
    logger.report_scalar(title="metrics", series="accuracy", value=metrics["accuracy"], iteration=0)
    logger.report_scalar(title="metrics", series="f1", value=metrics["f1"], iteration=0)
    logger.report_image(
        title="confusion_matrix",
        series="validation",
        iteration=0,
        local_path=str(figure_path),
    )
    task.upload_artifact("model_joblib", artifact_object=str(model_path))
    task.upload_artifact("metrics_json", artifact_object=metrics)

    from clearml import OutputModel

    output_model = OutputModel(
        task=task,
        framework="scikit-learn",
        name="sentiment_sklearn_pipeline",
    )
    output_model.update_weights(weights_filename=str(model_path))
    if hasattr(output_model, "set_metadata"):
        output_model.set_metadata("labels", json.dumps(LABELS))
        output_model.set_metadata("metrics", json.dumps(metrics))
    if hasattr(output_model, "set_tags"):
        output_model.set_tags(["course-project", "sentiment", "sklearn"])
    return output_model.id


def main() -> None:
    args = parse_args()
    task = init_clearml(args)
    dataset_path = resolve_dataset_path(args, task)
    data = load_dataset(dataset_path)

    train_df, valid_df = train_test_split(
        data,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=data["label"],
    )
    model = build_pipeline(
        classifier=args.classifier,
        c=args.c,
        ngram_max=args.ngram_max,
        max_features=args.max_features,
    )
    model.fit(train_df["text"], train_df["label"])
    predictions = model.predict(valid_df["text"])

    metrics = {
        "accuracy": float(accuracy_score(valid_df["label"], predictions)),
        "f1": float(f1_score(valid_df["label"], predictions, pos_label="positive")),
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
    }

    matrix = confusion_matrix(valid_df["label"], predictions, labels=LABELS)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{args.experiment_name}.joblib"
    metrics_path = output_dir / f"{args.experiment_name}.metrics.json"
    figure_path = output_dir / f"{args.experiment_name}.confusion_matrix.png"
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_matrix_image(matrix, figure_path)

    output_model_id = log_clearml_results(task, metrics, model_path, figure_path)
    if task is not None:
        task.close()

    print(json.dumps({"metrics": metrics, "model_path": str(model_path), "clearml_model_id": output_model_id}, indent=2))


if __name__ == "__main__":
    main()
