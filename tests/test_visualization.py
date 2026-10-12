"""Tests for the shared Matplotlib plotting utilities.

Every plot_* function must return a Figure (for inline `display(fig)` use in
notebooks) and, when given an out_path, must also save a PNG there so the
CLI scripts and results/plots/ artifacts keep working unchanged.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.visualization import (
    plot_actual_vs_predicted,
    plot_confusion_matrices,
    plot_customer_clusters,
    plot_dataset_overview,
    plot_elbow_and_silhouette,
    plot_model_comparison,
)


@pytest.fixture
def metrics_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["Logistic Regression", "KNN"],
            "accuracy": [0.81, 0.90],
            "precision": [0.82, 0.84],
            "recall": [0.62, 0.90],
            "f1_score": [0.71, 0.87],
        }
    )


def test_plot_model_comparison_returns_figure_without_saving(metrics_df: pd.DataFrame, tmp_path: Path) -> None:
    fig = plot_model_comparison(metrics_df, "Test comparison", out_path=None)
    assert isinstance(fig, Figure)
    assert len(list(tmp_path.iterdir())) == 0  # nothing written when out_path is None


def test_plot_model_comparison_saves_png_when_out_path_given(metrics_df: pd.DataFrame, tmp_path: Path) -> None:
    out_path = tmp_path / "comparison.png"
    fig = plot_model_comparison(metrics_df, "Test comparison", out_path=out_path)
    assert isinstance(fig, Figure)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_confusion_matrices_returns_figure(tmp_path: Path) -> None:
    matrices = {
        "Model A": np.array([[10, 2], [3, 15]]),
        "Model B": np.array([[12, 0], [1, 17]]),
    }
    out_path = tmp_path / "cm.png"
    fig = plot_confusion_matrices(matrices, out_path=out_path)
    assert isinstance(fig, Figure)
    assert out_path.exists()


def test_plot_confusion_matrices_without_out_path_does_not_save(tmp_path: Path) -> None:
    matrices = {"Model A": np.array([[10, 2], [3, 15]])}
    fig = plot_confusion_matrices(matrices)
    assert isinstance(fig, Figure)
    assert len(list(tmp_path.iterdir())) == 0


def test_plot_elbow_and_silhouette_returns_figure(tmp_path: Path) -> None:
    out_path = tmp_path / "elbow.png"
    fig = plot_elbow_and_silhouette(
        k_values=[2, 3, 4, 5],
        inertias=[100.0, 70.0, 50.0, 40.0],
        silhouettes=[0.4, 0.5, None, 0.45],
        out_path=out_path,
    )
    assert isinstance(fig, Figure)
    assert out_path.exists()


def test_plot_customer_clusters_returns_figure(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "Annual Income (k$)": [15, 16, 80, 85],
            "Spending Score (1-100)": [39, 81, 10, 90],
            "KMeans_cluster": [0, 0, 1, 1],
        }
    )
    out_path = tmp_path / "clusters.png"
    fig = plot_customer_clusters(
        df, "Annual Income (k$)", "Spending Score (1-100)", {"K-Means": "KMeans_cluster"}, out_path=out_path,
    )
    assert isinstance(fig, Figure)
    assert out_path.exists()


def test_plot_actual_vs_predicted_returns_figure(tmp_path: Path) -> None:
    results = {
        "Linear Regression": (np.array([1.0, 2.0, 3.0]), np.array([1.1, 1.9, 3.2])),
        "Gradient Boosting": (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])),
    }
    out_path = tmp_path / "actual_vs_predicted.png"
    fig = plot_actual_vs_predicted(results, out_path=out_path)
    assert isinstance(fig, Figure)
    assert out_path.exists()


def test_plot_dataset_overview_returns_figure(tmp_path: Path) -> None:
    social_df = pd.DataFrame(
        {"Purchased": [0, 1, 0, 1], "Age": [20, 30, 40, 50], "EstimatedSalary": [20000, 30000, 40000, 50000]}
    )
    mall_df = pd.DataFrame({"Annual Income (k$)": [15, 16, 80], "Spending Score (1-100)": [39, 81, 10]})
    insurance_df = pd.DataFrame({"charges": [1000.0, 2000.0, 3000.0], "smoker": ["no", "yes", "no"]})

    out_path = tmp_path / "overview.png"
    fig = plot_dataset_overview(social_df, mall_df, insurance_df, out_path=out_path)
    assert isinstance(fig, Figure)
    assert out_path.exists()
