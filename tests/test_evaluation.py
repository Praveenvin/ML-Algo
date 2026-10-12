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


def test_evaluate_classifier_multiclass_uses_weighted_averaging() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])
    y_proba = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.6, 0.3],
            [0.7, 0.2, 0.1],
            [0.1, 0.2, 0.7],
            [0.1, 0.1, 0.8],
        ]
    )
    metrics = evaluate_classifier(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == 4 / 6
    assert metrics["roc_auc"] is not None
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_evaluate_classifier_multiclass_without_full_proba_matrix_skips_roc_auc() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])
    # Only 2 columns given for a 3-class problem: shape mismatch, ROC-AUC should be skipped.
    y_proba = np.array([[0.8, 0.2]] * 6)
    metrics = evaluate_classifier(y_true, y_pred, y_proba)
    assert metrics["roc_auc"] is None


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


def test_compute_confusion_matrix_binary_with_missing_class_in_split() -> None:
    # Simulates a held-out split/model where class 1 never appears in y_true
    # or y_pred, even though the full dataset is binary (0/1). Without an
    # explicit `labels` list, sklearn would return a 1x1 matrix here, which
    # mismatches a 2-element class_labels list passed to the Plotly heatmap
    # and raises "length of the y vector must match img matrix" in px.imshow.
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    full_dataset_classes = [0, 1]

    cm = compute_confusion_matrix(y_true, y_pred, labels=full_dataset_classes)
    assert cm.shape == (2, 2)
    assert cm.sum() == 4
    assert cm[1, :].sum() == 0  # no actual class-1 samples in this split
    assert cm[:, 1].sum() == 0  # no predicted class-1 samples either


def test_compute_confusion_matrix_multiclass_with_class_absent_from_predictions() -> None:
    # 3-class problem where a model never predicts class 2 for any test
    # sample. The matrix must still be 3x3 (indexed by the full label set),
    # not 2x2 (indexed only by labels that appear in y_pred).
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 1])  # class 2 never predicted
    full_dataset_classes = [0, 1, 2]

    cm = compute_confusion_matrix(y_true, y_pred, labels=full_dataset_classes)
    assert cm.shape == (3, 3)
    assert cm.sum() == 6
    assert cm[:, 2].sum() == 0  # column for predicted class 2 is all zeros


def test_compute_confusion_matrix_without_labels_matches_sklearn_default() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm.shape == (3, 3)
    assert np.array_equal(cm, np.eye(3, dtype=cm.dtype) * 2)


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
