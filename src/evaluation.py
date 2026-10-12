"""Shared evaluation utilities for classification, clustering, and regression."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute standard classification metrics for binary or multiclass tasks.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities, if available, used to compute
            ROC-AUC. For binary tasks this may be either the full
            (n_samples, 2) probability matrix or just the positive-class
            column. For multiclass tasks it must be the full
            (n_samples, n_classes) probability matrix.

    Returns:
        Dictionary of metric name to value. Precision/recall/F1 use binary
        averaging for two-class problems and weighted averaging for
        multiclass problems. roc_auc is None when probabilities are
        unavailable or undefined (e.g. a single class present in y_true).
    """
    n_classes = len(np.unique(y_true))
    average = "binary" if n_classes == 2 else "weighted"

    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    roc_auc = None
    if y_proba is not None:
        try:
            if n_classes == 2:
                proba = y_proba[:, 1] if np.ndim(y_proba) == 2 else y_proba
                roc_auc = roc_auc_score(y_true, proba)
            elif np.ndim(y_proba) == 2 and y_proba.shape[1] == n_classes:
                roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        except ValueError:
            roc_auc = None
    metrics["roc_auc"] = roc_auc

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[Any] | None = None
) -> np.ndarray:
    """Return the confusion matrix for the given true/predicted labels.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Explicit, ordered label set to index the matrix by. Always
            pass the full set of classes from the whole dataset (not just
            the test split) so the matrix has a fixed shape regardless of
            whether every class happens to appear in this particular split
            or in a given model's predictions. Without this, sklearn infers
            labels from whatever values are present in y_true/y_pred, which
            can silently produce a smaller matrix than the caller expects
            (e.g. a 2x2 matrix for a 3-class problem where one class was
            never predicted), breaking any UI that assumes a fixed class list.

    Returns:
        A square matrix of shape (len(labels), len(labels)) when `labels` is
        given, otherwise sklearn's default behavior (shape inferred from the
        union of values seen in y_true and y_pred).
    """
    return confusion_matrix(y_true, y_pred, labels=labels)


def evaluate_clustering(
    X: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    """Compute clustering evaluation metrics, handling degenerate cases.

    Args:
        X: Feature matrix used for clustering (scaled).
        labels: Cluster assignment for each sample. Noise points from
            density-based algorithms should be labeled -1.

    Returns:
        Dictionary with number of clusters, number of noise points, and
        silhouette score (None if undefined, e.g. fewer than 2 valid clusters).
    """
    unique_labels = set(labels)
    n_noise = int(np.sum(labels == -1))
    n_clusters = len(unique_labels - {-1})

    silhouette = None
    valid_mask = labels != -1
    n_valid_clusters = len(set(labels[valid_mask])) if valid_mask.any() else 0
    if n_valid_clusters >= 2 and valid_mask.sum() > n_valid_clusters:
        try:
            silhouette = silhouette_score(X[valid_mask], labels[valid_mask])
        except ValueError:
            silhouette = None

    return {
        "n_clusters": n_clusters,
        "n_noise_points": n_noise,
        "silhouette_score": silhouette,
    }


def evaluate_regressor(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute MAE, RMSE, and R² for a regression model."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y_true, y_pred),
    }
