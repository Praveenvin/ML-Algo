"""Shared plotting utilities. All functions save a PNG to the given path."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_dataset_overview(
    social_df: pd.DataFrame, mall_df: pd.DataFrame, insurance_df: pd.DataFrame, out_path: Path
) -> None:
    """Plot a quick overview grid summarizing the three raw datasets."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    sns.countplot(x="Purchased", data=social_df, ax=axes[0, 0])
    axes[0, 0].set_title("Social Network Ads: Purchased distribution")

    sns.histplot(social_df["Age"], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title("Social Network Ads: Age distribution")

    sns.histplot(social_df["EstimatedSalary"], kde=True, ax=axes[0, 2])
    axes[0, 2].set_title("Social Network Ads: Estimated Salary distribution")

    sns.scatterplot(
        x="Annual Income (k$)", y="Spending Score (1-100)", data=mall_df, ax=axes[1, 0]
    )
    axes[1, 0].set_title("Mall Customers: Income vs Spending Score")

    sns.histplot(insurance_df["charges"], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title("Insurance: Charges distribution")

    sns.boxplot(x="smoker", y="charges", data=insurance_df, ax=axes[1, 2])
    axes[1, 2].set_title("Insurance: Charges by smoker status")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_model_comparison(metrics_df: pd.DataFrame, title: str, out_path: Path) -> None:
    """Bar chart comparing models across accuracy, precision, recall, f1_score.

    Args:
        metrics_df: DataFrame with a 'model' column and metric columns.
        title: Chart title.
        out_path: Where to save the PNG.
    """
    plot_metrics = [m for m in ["accuracy", "precision", "recall", "f1_score"] if m in metrics_df.columns]
    melted = metrics_df.melt(id_vars="model", value_vars=plot_metrics, var_name="metric", value_name="score")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="model", y="score", hue="metric", data=melted, ax=ax)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(
    matrices: dict[str, np.ndarray], out_path: Path
) -> None:
    """Plot a grid of confusion matrix heatmaps, one per model."""
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

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_elbow_and_silhouette(
    k_values: list[int], inertias: list[float], silhouettes: list[float | None], out_path: Path
) -> None:
    """Plot the K-Means elbow curve and silhouette scores side by side."""
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

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_customer_clusters(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_cols: dict[str, str],
    out_path: Path,
) -> None:
    """Plot labeled scatter plots of customer clusters for each algorithm.

    Args:
        df: DataFrame containing feature columns and one label column per algorithm.
        x_col: Column name for the x-axis.
        y_col: Column name for the y-axis.
        label_cols: Mapping of subplot title to the column holding its cluster labels.
        out_path: Where to save the PNG.
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

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_actual_vs_predicted(
    results: dict[str, tuple[np.ndarray, np.ndarray]], out_path: Path
) -> None:
    """Plot actual vs predicted scatter plots for each regression model.

    Args:
        results: Mapping of model name to (y_true, y_pred) arrays.
        out_path: Where to save the PNG.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    axes = np.atleast_1d(axes)

    for ax, (name, (y_true, y_pred)) in zip(axes, results.items()):
        ax.scatter(y_true, y_pred, alpha=0.5, s=15)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("Actual charges")
        ax.set_ylabel("Predicted charges")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
