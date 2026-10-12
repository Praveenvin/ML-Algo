"""Persist and reload interactive-app experiment results.

Unlike the CLI modules (which overwrite one fixed CSV per task), the app can
run many experiments across many datasets in one session. Each call to
`save_experiment` writes a uniquely named metrics CSV and appends a row to a
single index file, so results from different runs never clobber each other
and the Model Comparison page can list/reload everything that has run.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src import config

EXPERIMENT_INDEX_PATH = config.METRICS_DIR / "experiment_index.csv"

INDEX_COLUMNS = [
    "experiment_id",
    "task",
    "dataset_name",
    "target_column",
    "timestamp",
    "best_model",
    "selection_metric",
    "metrics_path",
]


def save_experiment(
    task: str,
    dataset_name: str,
    metrics_df: pd.DataFrame,
    best_model: str | None = None,
    selection_metric: str | None = None,
    target_column: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Save one experiment's metrics to a uniquely named CSV and index it.

    Args:
        task: One of "classification", "clustering", "boosting", "regression".
        dataset_name: Human-readable dataset label (sample name or uploaded filename).
        metrics_df: Per-model metrics table to persist.
        best_model: Name of the selected best model, if applicable.
        selection_metric: Description of how the best model was chosen.
        target_column: Target column name, if applicable (None for clustering).
        extra: Additional JSON-serializable metadata to store alongside the index row.

    Returns:
        The generated experiment_id.
    """
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    experiment_id = uuid.uuid4().hex[:12]
    timestamp = datetime.now(timezone.utc).isoformat()
    safe_dataset_name = "".join(c if c.isalnum() else "_" for c in dataset_name)
    metrics_filename = f"experiment_{task}_{safe_dataset_name}_{experiment_id}.csv"
    metrics_path = config.METRICS_DIR / metrics_filename

    metrics_df.to_csv(metrics_path, index=False)

    row = {
        "experiment_id": experiment_id,
        "task": task,
        "dataset_name": dataset_name,
        "target_column": target_column or "",
        "timestamp": timestamp,
        "best_model": best_model or "",
        "selection_metric": selection_metric or "",
        "metrics_path": str(metrics_path.relative_to(config.PROJECT_ROOT)),
        "extra_json": json.dumps(extra or {}),
    }

    if EXPERIMENT_INDEX_PATH.exists():
        index_df = pd.read_csv(EXPERIMENT_INDEX_PATH)
        index_df = pd.concat([index_df, pd.DataFrame([row])], ignore_index=True)
    else:
        index_df = pd.DataFrame([row])

    index_df.to_csv(EXPERIMENT_INDEX_PATH, index=False)
    return experiment_id


def load_experiment_index() -> pd.DataFrame:
    """Load the experiment index, or an empty frame with the expected columns."""
    if not EXPERIMENT_INDEX_PATH.exists():
        return pd.DataFrame(columns=INDEX_COLUMNS + ["extra_json"])
    return pd.read_csv(EXPERIMENT_INDEX_PATH)


def load_experiment_metrics(experiment_id: str) -> pd.DataFrame:
    """Load a single experiment's metrics table by its experiment_id.

    Raises:
        ValueError: If no experiment with that id is indexed.
    """
    index_df = load_experiment_index()
    match = index_df[index_df["experiment_id"] == experiment_id]
    if match.empty:
        raise ValueError(f"No experiment found with id '{experiment_id}'.")
    metrics_path = config.PROJECT_ROOT / match.iloc[0]["metrics_path"]
    return pd.read_csv(metrics_path)


def load_all_results(task: str | None = None) -> pd.DataFrame:
    """Load and concatenate metrics for every indexed experiment, tagged with
    their dataset/experiment metadata, optionally filtered by task.

    Returns:
        A combined DataFrame with one row per (experiment, model) pair, or an
        empty DataFrame if no experiments are indexed yet.
    """
    index_df = load_experiment_index()
    if task is not None:
        index_df = index_df[index_df["task"] == task]

    frames = []
    for _, row in index_df.iterrows():
        metrics_path = config.PROJECT_ROOT / row["metrics_path"]
        if not metrics_path.exists():
            continue
        metrics_df = pd.read_csv(metrics_path)
        metrics_df.insert(0, "experiment_id", row["experiment_id"])
        metrics_df.insert(1, "task", row["task"])
        metrics_df.insert(2, "dataset_name", row["dataset_name"])
        metrics_df.insert(3, "timestamp", row["timestamp"])
        frames.append(metrics_df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
