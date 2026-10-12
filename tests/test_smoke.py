"""Smoke tests for the full training pipelines and output file generation."""

from src import config
from src.boosting import run_boosting
from src.classification import run_classification
from src.clustering import run_clustering
from src.regression import run_regression


def test_classification_pipeline_runs_and_writes_outputs() -> None:
    result = run_classification()
    assert len(result["metrics_df"]) == 5
    assert result["best_model_name"] in result["metrics_df"]["model"].values
    assert (config.METRICS_DIR / "classification_metrics.csv").exists()
    assert (config.PREDICTIONS_DIR / "classification_predictions.csv").exists()
    assert (config.MODELS_DIR / "best_classification_model.joblib").exists()


def test_clustering_pipeline_runs_and_writes_outputs() -> None:
    result = run_clustering()
    assert len(result["metrics_df"]) == 3
    assert (config.METRICS_DIR / "clustering_metrics.csv").exists()
    assert (config.CLUSTER_ASSIGNMENTS_DIR / "customer_clusters.csv").exists()


def test_boosting_pipeline_runs_and_writes_outputs() -> None:
    result = run_boosting()
    assert len(result["metrics_df"]) == 3
    assert (config.METRICS_DIR / "boosting_metrics.csv").exists()
    assert (config.PREDICTIONS_DIR / "boosting_predictions.csv").exists()
    assert (config.MODELS_DIR / "best_boosting_model.joblib").exists()


def test_regression_pipeline_runs_and_writes_outputs() -> None:
    result = run_regression()
    assert len(result["metrics_df"]) == 4
    assert (config.METRICS_DIR / "regression_metrics.csv").exists()
    assert (config.PREDICTIONS_DIR / "regression_predictions.csv").exists()
    assert (config.MODELS_DIR / "best_regression_model.joblib").exists()
