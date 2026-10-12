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
    """Compute standard classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probability of the positive class, if available,
            used to compute ROC-AUC.

    Returns:
        Dictionary of metric name to value. roc_auc is None when probabilities
        are unavailable or undefined (e.g. a single class present).
    """
    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    roc_auc = None
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            roc_auc = None
    metrics["roc_auc"] = roc_auc

    return metrics


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return the confusion matrix for the given true/predicted labels."""
    return confusion_matrix(y_true, y_pred)


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
