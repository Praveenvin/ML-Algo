"""Tests for evaluation metrics, including clustering edge cases."""

import numpy as np

from src.evaluation import (
    compute_confusion_matrix,
    evaluate_classifier,
    evaluate_clustering,
    evaluate_regressor,
)


def test_evaluate_classifier_perfect_predictions() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8])
    metrics = evaluate_classifier(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_evaluate_classifier_without_proba_has_none_roc_auc() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    metrics = evaluate_classifier(y_true, y_pred, None)
    assert metrics["roc_auc"] is None


def test_compute_confusion_matrix_shape() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm.shape == (2, 2)
    assert cm.sum() == 4


def test_evaluate_clustering_normal_case() -> None:
    X = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=float)
    labels = np.array([0, 0, 1, 1])
    metrics = evaluate_clustering(X, labels)
    assert metrics["n_clusters"] == 2
    assert metrics["n_noise_points"] == 0
    assert metrics["silhouette_score"] is not None


def test_evaluate_clustering_single_cluster_returns_none_silhouette() -> None:
    X = np.array([[0, 0], [0, 1], [1, 0]], dtype=float)
    labels = np.array([0, 0, 0])
    metrics = evaluate_clustering(X, labels)
    assert metrics["n_clusters"] == 1
    assert metrics["silhouette_score"] is None


def test_evaluate_clustering_all_noise_returns_none_silhouette() -> None:
    X = np.array([[0, 0], [5, 5], [10, 10]], dtype=float)
    labels = np.array([-1, -1, -1])
    metrics = evaluate_clustering(X, labels)
    assert metrics["n_clusters"] == 0
    assert metrics["n_noise_points"] == 3
    assert metrics["silhouette_score"] is None


def test_evaluate_regressor_perfect_predictions() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    metrics = evaluate_regressor(y_true, y_pred)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0
