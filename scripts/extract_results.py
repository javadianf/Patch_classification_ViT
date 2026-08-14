# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Javadian. All rights reserved.

"""
Flatten the MLflow store into a single results CSV.

Reads the SQLite backend directly rather than going through the MLflow client,
because the per-run evaluation JSON is an artifact on disk and pairing it with
the run parameters is a join the client API does not do in one pass.
"""

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd

MLRUNS_PATH = Path(os.environ.get("CCRCC_MLRUNS", Path.cwd() / "mlruns"))
OUTPUT_FILE = Path(
    os.environ.get("CCRCC_RESULTS_CSV", Path.cwd() / "mlflow_results.csv")
)


def load_tables(db_path):
    conn = sqlite3.connect(db_path)
    try:
        params_df = pd.read_sql_query("SELECT run_uuid, key, value FROM params", conn)
        metrics_df = pd.read_sql_query(
            "SELECT run_uuid, key, value, step FROM metrics", conn
        )
        runs_df = pd.read_sql_query(
            "SELECT run_uuid, experiment_id, status FROM runs "
            "WHERE status = 'FINISHED'",
            conn,
        )
    finally:
        conn.close()

    return params_df, metrics_df, runs_df


def last_metric_value(run_metrics, key):
    """Take the value at the highest step, which is the end-of-training value."""
    subset = run_metrics[run_metrics["key"] == key]
    if subset.empty:
        return None
    return subset.loc[subset["step"].idxmax(), "value"]


def extract_experiment_data(mlruns_path=MLRUNS_PATH, output_file=OUTPUT_FILE):
    if not mlruns_path.exists():
        print(f"mlruns directory not found: {mlruns_path}")
        return pd.DataFrame()

    db_path = mlruns_path / "mlflow.db"
    if not db_path.exists():
        print(f"MLflow database not found: {db_path}")
        return pd.DataFrame()

    params_df, metrics_df, runs_df = load_tables(db_path)
    results = []

    for _, run_row in runs_df.iterrows():
        run_uuid = run_row["run_uuid"]
        experiment_id = run_row["experiment_id"]

        run_params = params_df[params_df["run_uuid"] == run_uuid]
        params_dict = dict(zip(run_params["key"], run_params["value"]))

        run_metrics = metrics_df[metrics_df["run_uuid"] == run_uuid]

        run_dir = mlruns_path / str(experiment_id) / run_uuid
        metrics_dir = run_dir / "artifacts" / "metrics"
        if not metrics_dir.exists():
            continue

        evaluation_files = list(metrics_dir.glob("*_evaluation.json"))
        if not evaluation_files:
            continue

        metrics_file = evaluation_files[0]
        dataform = metrics_file.stem.replace("_evaluation", "")

        with open(metrics_file, "r") as f:
            metrics_data = json.load(f)

        val_metrics = metrics_data.get("validation", {})
        test_metrics = metrics_data.get("test", {})

        results.append(
            {
                "experiment_id": experiment_id,
                "run_id": run_uuid,
                "dataform": dataform,
                "model_name": params_dict.get("model_name"),
                "random_seed": params_dict.get("random_seed"),
                "apply_flip_augmentation": params_dict.get("apply_flip_augmentation"),
                "perturbation_type": params_dict.get("perturbation_type"),
                "perturbation_level": params_dict.get("perturbation_level"),
                "num_epochs": params_dict.get("num_epochs"),
                "mixer_configs": params_dict.get("mixer_configs"),
                "config_name": params_dict.get("config_name"),
                "train_accuracy": last_metric_value(run_metrics, "train_accuracy"),
                "train_balanced_accuracy": last_metric_value(
                    run_metrics, "train_balanced_accuracy"
                ),
                "val_accuracy": val_metrics.get("accuracy"),
                "val_balanced_accuracy": val_metrics.get("balanced_accuracy"),
                "val_precision": val_metrics.get("precision"),
                "val_recall": val_metrics.get("recall"),
                "val_f1": val_metrics.get("f1"),
                "test_accuracy": test_metrics.get("accuracy"),
                "test_balanced_accuracy": test_metrics.get("balanced_accuracy"),
                "test_precision": test_metrics.get("precision"),
                "test_recall": test_metrics.get("recall"),
                "test_f1": test_metrics.get("f1"),
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    print(f"Wrote {len(results)} runs to {output_file}")

    return df


if __name__ == "__main__":
    df = extract_experiment_data()
    print(df.head())
    print(f"Shape: {df.shape}")
