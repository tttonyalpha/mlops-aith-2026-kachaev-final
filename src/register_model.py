from __future__ import annotations

import argparse
import json

from clearml import Model, Task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a ClearML model to Model Registry.")
    parser.add_argument("--project", default="mlops-aith-2026-kachaev")
    parser.add_argument("--model-id", required=True, help="ClearML OutputModel id printed by src/train.py.")
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--tags", nargs="*", default=["best", "registry", "sentiment", "course-project"])
    parser.add_argument("--metrics-json", default=None, help="Optional metrics file produced by train.py.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = Task.init(
        project_name=args.project,
        task_name=f"publish registry model {args.model_id}",
        task_type=Task.TaskTypes.data_processing,
    )
    task.connect(vars(args), name="registry_params")

    model = Model(model_id=args.model_id)
    if hasattr(model, "set_tags"):
        model.set_tags(args.tags)
    if hasattr(model, "set_metadata"):
        model.set_metadata("version", args.version)
    if args.metrics_json:
        with open(args.metrics_json, "r", encoding="utf-8") as fh:
            if hasattr(model, "set_metadata"):
                model.set_metadata("metrics", json.dumps(json.load(fh)))

    model.publish()
    task.upload_artifact("published_model_id", artifact_object=args.model_id)
    task.close()
    print(f"Published model to ClearML Model Registry: {args.model_id}")


if __name__ == "__main__":
    main()
