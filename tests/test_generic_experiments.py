"""Tests for the generic, app-facing experiment runners (arbitrary dataset/
feature/target selection, as opposed to the CLI modules' hardcoded columns)
and their input validation."""

import pandas as pd
import pytest

from src.boosting import run_boosting_experiment
from src.classification import run_classification_experiment
from src.clustering import run_clustering_experiment
from src.data_loader import load_mall_customers, load_social_network_ads
from src.regression import run_regression_experiment

SOCIAL_FEATURES = ["Age", "EstimatedSalary", "Gender"]
MALL_FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]


def _social_df() -> pd.DataFrame:
    from src import config

    return load_social_network_ads(config.SOCIAL_NETWORK_ADS_PATH)


def _mall_df() -> pd.DataFrame:
    from src import config

    return load_mall_customers(config.MALL_CUSTOMERS_PATH)


def test_run_classification_experiment_basic() -> None:
    df = _social_df()
    result = run_classification_experiment(
        df, SOCIAL_FEATURES, "Purchased", model_names=["Logistic Regression", "KNN"],
        test_size=0.2, random_state=42,
    )
    assert set(result["metrics_df"]["model"]) == {"Logistic Regression", "KNN"}
    assert result["best_model_name"] in {"Logistic Regression", "KNN"}
    assert "Logistic Regression" in result["probabilities"]
    assert result["n_classes"] == 2


def test_run_classification_experiment_confusion_matrices_match_full_class_list() -> None:
    """Regression test for a bug where a class entirely absent from the
    held-out test split (and thus from both y_true and every model's
    y_pred) produced a smaller confusion matrix than result["classes"],
    which broke the Plotly heatmap in ui/components.py.plot_confusion_matrix
    (ValueError: length of the y vector must match the img matrix).

    This happens when a class is too rare to stratify on (count < 2): the
    split falls back to a non-stratified random split, and a singleton class
    can land entirely in the training fold, leaving the test fold with fewer
    classes than the full dataset.
    """
    import numpy as np

    rng = np.random.default_rng(1)
    rows = []
    for _ in range(30):
        rows.append({"x1": rng.normal(0, 1), "x2": rng.normal(0, 1), "label": 0})
    for _ in range(30):
        rows.append({"x1": rng.normal(5, 1), "x2": rng.normal(3, 1), "label": 1})
    rows.append({"x1": 20.0, "x2": 20.0, "label": 2})  # single rare-class sample
    df = pd.DataFrame(rows)

    result = run_classification_experiment(
        df, ["x1", "x2"], "label", model_names=["Logistic Regression", "Decision Tree", "KNN"],
        test_size=0.3, random_state=42,
    )

    # Confirms this test actually exercises the bug scenario: class 2 is
    # absent from the test split entirely (not just from predictions).
    assert 2 not in set(result["y_test"])
    assert len(result["classes"]) == 3

    expected_n = len(result["classes"])
    for model_name, cm in result["confusion_matrices"].items():
        assert cm.shape == (expected_n, expected_n), (
            f"{model_name}: confusion matrix shape {cm.shape} does not match "
            f"the {expected_n} classes in result['classes']"
        )


def test_run_classification_experiment_rejects_empty_features() -> None:
    df = _social_df()
    with pytest.raises(ValueError, match="feature column"):
        run_classification_experiment(df, [], "Purchased")


def test_run_classification_experiment_rejects_empty_model_list() -> None:
    df = _social_df()
    with pytest.raises(ValueError, match="algorithm"):
        run_classification_experiment(df, SOCIAL_FEATURES, "Purchased", model_names=[])


def test_run_classification_experiment_rejects_single_class_target() -> None:
    df = _social_df().copy()
    df["Purchased"] = 0
    with pytest.raises(ValueError, match="unique value"):
        run_classification_experiment(df, SOCIAL_FEATURES, "Purchased")


def test_run_classification_experiment_rejects_unknown_model_name() -> None:
    df = _social_df()
    with pytest.raises(ValueError, match="Unknown algorithm"):
        run_classification_experiment(df, SOCIAL_FEATURES, "Purchased", model_names=["Not A Real Model"])


def test_run_boosting_experiment_basic() -> None:
    df = _social_df()
    result = run_boosting_experiment(
        df, SOCIAL_FEATURES, "Purchased", model_names=["AdaBoost"], test_size=0.2, random_state=42,
    )
    assert list(result["metrics_df"]["model"]) == ["AdaBoost"]
    assert result["feature_importances"]["AdaBoost"] is not None


def test_run_regression_experiment_basic() -> None:
    df = pd.DataFrame(
        {
            "x1": range(50),
            "x2": [v * 2 for v in range(50)],
            "y": [v * 3 + 1 for v in range(50)],
        }
    )
    result = run_regression_experiment(
        df, ["x1", "x2"], "y", model_names=["Linear Regression"], test_size=0.2, random_state=42,
    )
    assert list(result["metrics_df"]["model"]) == ["Linear Regression"]
    assert result["metrics_df"]["r2"].iloc[0] > 0.9


def test_run_regression_experiment_rejects_non_numeric_target() -> None:
    df = pd.DataFrame({"x1": [1, 2, 3], "label": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="not numeric"):
        run_regression_experiment(df, ["x1"], "label")


def test_run_clustering_experiment_basic() -> None:
    df = _mall_df()
    result = run_clustering_experiment(
        df, MALL_FEATURES, algorithms=["KMeans"], kmeans_k=5, random_state=42,
    )
    assert list(result["metrics_df"]["model"]) == ["KMeans"]
    assert "KMeans_cluster" in result["assignments_df"].columns
    assert result["pca_df"] is None  # only 2 features selected, no PCA needed


def test_run_clustering_experiment_pca_for_more_than_two_features() -> None:
    df = _mall_df()
    result = run_clustering_experiment(
        df, MALL_FEATURES + ["Age"], algorithms=["KMeans"], kmeans_k=5, random_state=42,
    )
    assert result["pca_df"] is not None
    assert list(result["pca_df"].columns) == ["PC1", "PC2"]


def test_run_clustering_experiment_rejects_fewer_than_two_numeric_features() -> None:
    df = _mall_df()
    with pytest.raises(ValueError, match="at least 2 numeric"):
        run_clustering_experiment(df, ["Genre"], algorithms=["KMeans"])


def test_run_clustering_experiment_rejects_unknown_algorithm() -> None:
    df = _mall_df()
    with pytest.raises(ValueError, match="Unknown algorithm"):
        run_clustering_experiment(df, MALL_FEATURES, algorithms=["NotAnAlgorithm"])
