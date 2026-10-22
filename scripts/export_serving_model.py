"""Explicitly promote a local MLflow run to the small serving bundle."""
import argparse
import hashlib
import json
from pathlib import Path

import mlflow.sklearn
from mlflow.tracking import MlflowClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    mlflow.set_tracking_uri((root / "mlruns").as_uri())
    client = MlflowClient()
    run = client.get_run(args.run_id)
    if run.info.status != "FINISHED":
        raise ValueError("Only a FINISHED run can be exported")
    model = mlflow.sklearn.load_model(f"runs:/{args.run_id}/model")
    columns_path = client.download_artifacts(args.run_id, "feature_columns.txt")
    columns = Path(columns_path).read_text().splitlines()
    if columns != model.get_booster().feature_names:
        raise ValueError("Model and feature metadata do not match")
    threshold = float(run.data.params["threshold"])
    if not 0 < threshold < 1:
        raise ValueError("Invalid classification threshold")
    destination = root / "src" / "serving" / "bundle"
    destination.mkdir(parents=True, exist_ok=True)
    model.save_model(destination / "model.ubj")
    manifest = {
        "schema_version": 1, "run_id": args.run_id,
        "threshold": threshold, "feature_columns": columns,
        "model_sha256": hashlib.sha256((destination / "model.ubj").read_bytes()).hexdigest(),
        "metrics": run.data.metrics,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported run {args.run_id} to {destination}")


if __name__ == "__main__":
    main()
