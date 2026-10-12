"""Shared plotting utilities.

Every function here builds a Matplotlib `Figure` and returns it, so callers
(notebooks in particular) can display it inline with `display(fig)` without
round-tripping through a saved PNG. When an `out_path` is given, the figure
is also saved there (dpi=150), so the CLI scripts and existing results/plots/
artifacts are unaffected.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

sns.set_theme(style="whitegrid")


def _finalize(fig: Figure, out_path: Path | None) -> Figure:
    """Apply a tight layout, optionally save to disk, and return the figure.

    The figure is closed via `plt.close(fig)` after saving so that repeated
    calls (e.g. across scripts/run_all.py) don't accumulate hidden figures in
    pyplot's global state. This is safe for later display: a closed figure's
    canvas and artists remain intact, so `display(fig)` in a notebook still
    renders it correctly even though it's no longer tracked by pyplot.
    """
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return fig


def plot_dataset_overview(
    social_df: pd.DataFrame,
    mall_df: pd.DataFrame,
    insurance_df: pd.DataFrame,
    out_path: Path | None = None,
) -> Figure:
    """Plot a quick overview grid summarizing the three raw datasets."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    sns.countplot(x="Purchased", data=social_df, ax=axes[0, 0])
    axes[0, 0].set_title("Social Network Ads: Purchased distribution")
    axes[0, 0].set_xlabel("Purchased")
    axes[0, 0].set_ylabel("Count")

    sns.histplot(social_df["Age"], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title("Social Network Ads: Age distribution")
    axes[0, 1].set_xlabel("Age")
    axes[0, 1].set_ylabel("Count")

    sns.histplot(social_df["EstimatedSalary"], kde=True, ax=axes[0, 2])
    axes[0, 2].set_title("Social Network Ads: Estimated Salary distribution")
    axes[0, 2].set_xlabel("Estimated Salary")
    axes[0, 2].set_ylabel("Count")

    sns.scatterplot(
        x="Annual Income (k$)", y="Spending Score (1-100)", data=mall_df, ax=axes[1, 0]
    )
    axes[1, 0].set_title("Mall Customers: Income vs Spending Score")
    axes[1, 0].set_xlabel("Annual Income (k$)")
    axes[1, 0].set_ylabel("Spending Score (1-100)")

    sns.histplot(insurance_df["charges"], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title("Insurance: Charges distribution")
    axes[1, 1].set_xlabel("Charges")
    axes[1, 1].set_ylabel("Count")

    sns.boxplot(x="smoker", y="charges", data=insurance_df, ax=axes[1, 2])
    axes[1, 2].set_title("Insurance: Charges by smoker status")
    axes[1, 2].set_xlabel("Smoker")
    axes[1, 2].set_ylabel("Charges")

    return _finalize(fig, out_path)


def plot_model_comparison(metrics_df: pd.DataFrame, title: str, out_path: Path | None = None) -> Figure:
    """Bar chart comparing models across accuracy, precision, recall, f1_score.

    Args:
        metrics_df: DataFrame with a 'model' column and metric columns.
        title: Chart title.
        out_path: If given, also save the figure here (dpi=150).

    Returns:
        The Matplotlib Figure, for display via `display(fig)` or saving.
    """
    plot_metrics = [m for m in ["accuracy", "precision", "recall", "f1_score"] if m in metrics_df.columns]
    melted = metrics_df.melt(id_vars="model", value_vars=plot_metrics, var_name="metric", value_name="score")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="model", y="score", hue="metric", data=melted, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Metric", loc="lower right")

    return _finalize(fig, out_path)


def plot_confusion_matrices(matrices: dict[str, np.ndarray], out_path: Path | None = None) -> Figure:
    """Plot a grid of confusion matrix heatmaps, one per model.

    Args:
        matrices: Mapping of model name to confusion matrix.
        out_path: If given, also save the figure here (dpi=150).

    Returns:
        The Matplotlib Figure, for display via `display(fig)` or saving.
    """
    n = len(matrices)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (name, cm) in zip(axes, matrices.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    for ax in axes[len(matrices):]:
        ax.axis("off")

    return _finalize(fig, out_path)


def plot_elbow_and_silhouette(
    k_values: list[int],
    inertias: list[float],
    silhouettes: list[float | None],
    out_path: Path | None = None,
) -> Figure:
    """Plot the K-Means elbow curve and silhouette scores side by side.

    Returns:
        The Matplotlib Figure, for display via `display(fig)` or saving.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(k_values, inertias, marker="o")
    axes[0].set_title("Elbow Method (Inertia vs K)")
    axes[0].set_xlabel("Number of clusters (K)")
    axes[0].set_ylabel("Inertia")

    valid = [(k, s) for k, s in zip(k_values, silhouettes) if s is not None]
    if valid:
        xs, ys = zip(*valid)
        axes[1].plot(xs, ys, marker="o", color="darkorange")
    axes[1].set_title("Silhouette Score vs K")
    axes[1].set_xlabel("Number of clusters (K)")
    axes[1].set_ylabel("Silhouette score")

    return _finalize(fig, out_path)


def plot_customer_clusters(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_cols: dict[str, str],
    out_path: Path | None = None,
) -> Figure:
    """Plot labeled scatter plots of customer clusters for each algorithm.

    Args:
        df: DataFrame containing feature columns and one label column per algorithm.
        x_col: Column name for the x-axis.
        y_col: Column name for the y-axis.
        label_cols: Mapping of subplot title to the column holding its cluster labels.
        out_path: If given, also save the figure here (dpi=150).

    Returns:
        The Matplotlib Figure, for display via `display(fig)` or saving.
    """
    n = len(label_cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    axes = np.atleast_1d(axes)

    for ax, (title, col) in zip(axes, label_cols.items()):
        palette = sns.color_palette("tab10", df[col].nunique())
        sns.scatterplot(
            x=x_col, y=y_col, hue=col, data=df, palette=palette, ax=ax, legend="full"
        )
        ax.set_title(title)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend(title="Cluster", loc="best")

    return _finalize(fig, out_path)


def plot_actual_vs_predicted(
    results: dict[str, tuple[np.ndarray, np.ndarray]], out_path: Path | None = None
) -> Figure:
    """Plot actual vs predicted scatter plots for each regression model.

    Args:
        results: Mapping of model name to (y_true, y_pred) arrays.
        out_path: If given, also save the figure here (dpi=150).

    Returns:
        The Matplotlib Figure, for display via `display(fig)` or saving.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    axes = np.atleast_1d(axes)

    for ax, (name, (y_true, y_pred)) in zip(axes, results.items()):
        ax.scatter(y_true, y_pred, alpha=0.5, s=15, label="Predictions")
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
        ax.set_title(name)
        ax.set_xlabel("Actual charges")
        ax.set_ylabel("Predicted charges")
        ax.legend(loc="best")

    return _finalize(fig, out_path)
