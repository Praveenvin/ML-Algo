"""Reusable Streamlit + Plotly widgets shared across ML Studio pages.

Keeps UI rendering separate from ML logic: every function here either
renders a widget or builds a Plotly figure from already-computed results —
none of them train models or touch scikit-learn directly.
"""

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve

from ui.styles import PRIMARY, TEXT_MUTED

PLOTLY_TEMPLATE = "plotly_white"
CATEGORICAL_PALETTE = px.colors.qualitative.Prism


def metric_card(label: str, value: str, help_text: str | None = None) -> None:
    """Render a single metric card using st.metric for native formatting."""
    st.metric(label=label, value=value, help=help_text)


def empty_state(message: str, hint: str | None = None) -> None:
    """Render a friendly empty-state placeholder instead of a blank page."""
    hint_html = f'<p style="margin-top:0.4rem;font-size:0.85rem;">{hint}</p>' if hint else ""
    st.markdown(
        f"""
        <div class="ms-empty-state">
            <p style="font-weight:600;margin-bottom:0.2rem;">{message}</p>
            {hint_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str) -> None:
    st.markdown(f'<p class="ms-section-title">{title}</p>', unsafe_allow_html=True)


def dataframe_quality_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Compute the quality summary shown in the Dataset Explorer and Dashboard."""
    missing = df.isna().sum()
    return {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "n_missing": int(missing.sum()),
        "missing_by_column": missing[missing > 0].to_dict(),
        "n_duplicates": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def render_quality_warnings(summary: dict[str, Any]) -> None:
    """Render data-quality warnings based on a dataframe_quality_summary() result."""
    if summary["n_missing"] > 0:
        cols = ", ".join(f"{c} ({n})" for c, n in summary["missing_by_column"].items())
        st.warning(f"Missing values detected in: {cols}. Rows with missing values will be "
                   f"rejected by model training until resolved.")
    if summary["n_duplicates"] > 0:
        st.warning(f"{summary['n_duplicates']} duplicate row(s) detected.")
    if summary["n_missing"] == 0 and summary["n_duplicates"] == 0:
        st.success("No missing values or duplicate rows detected.")


def download_dataframe_button(df: pd.DataFrame, label: str, filename: str, key: str) -> None:
    """Render a download button for a DataFrame as CSV."""
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def plot_distribution(df: pd.DataFrame, column: str) -> go.Figure:
    """Build a histogram (numeric) or bar chart (categorical) for one column."""
    if pd.api.types.is_numeric_dtype(df[column]):
        fig = px.histogram(df, x=column, nbins=30, template=PLOTLY_TEMPLATE,
                            color_discrete_sequence=[PRIMARY])
    else:
        counts = df[column].value_counts().reset_index()
        counts.columns = [column, "count"]
        fig = px.bar(counts, x=column, y="count", template=PLOTLY_TEMPLATE,
                     color_discrete_sequence=[PRIMARY])
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=360)
    return fig


def plot_metric_comparison(metrics_df: pd.DataFrame, metrics_to_plot: list[str], title: str) -> go.Figure:
    """Grouped bar chart comparing models across the selected metric columns."""
    available = [m for m in metrics_to_plot if m in metrics_df.columns]
    melted = metrics_df.melt(id_vars="model", value_vars=available, var_name="metric", value_name="score")
    fig = px.bar(
        melted, x="model", y="score", color="metric", barmode="group",
        template=PLOTLY_TEMPLATE, title=title, color_discrete_sequence=CATEGORICAL_PALETTE,
    )
    fig.update_layout(yaxis_range=[0, 1], margin=dict(l=10, r=10, t=50, b=10), legend_title="")
    return fig


def plot_regression_comparison(metrics_df: pd.DataFrame, title: str) -> go.Figure:
    """Three side-by-side bar charts (MAE, RMSE, R²) comparing regression models.

    Unlike classification metrics, these are on different scales and R² can
    be negative, so each gets its own y-axis rather than a shared [0, 1] range.
    """
    fig = make_subplots(rows=1, cols=3, subplot_titles=("MAE (lower is better)",
                                                         "RMSE (lower is better)",
                                                         "R² (higher is better)"))
    for i, metric in enumerate(["mae", "rmse", "r2"], start=1):
        fig.add_trace(
            go.Bar(x=metrics_df["model"], y=metrics_df[metric], marker_color=PRIMARY, showlegend=False),
            row=1, col=i,
        )
    fig.update_layout(title=title, template=PLOTLY_TEMPLATE, margin=dict(l=10, r=10, t=70, b=10), height=380)
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_labels: list[str], title: str) -> go.Figure:
    """Heatmap of a single confusion matrix with annotated counts."""
    fig = px.imshow(
        cm, x=[f"Pred {c}" for c in class_labels], y=[f"Actual {c}" for c in class_labels],
        text_auto=True, color_continuous_scale=[[0, "#FFFFFF"], [1, PRIMARY]],
        template=PLOTLY_TEMPLATE, title=title,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), coloraxis_showscale=False)
    return fig


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, classes: list, title: str) -> go.Figure | None:
    """Build an ROC curve figure. Binary: single curve. Multiclass: one-vs-rest per class.

    Returns None if probabilities are unavailable or ROC-AUC cannot be computed.
    """
    fig = go.Figure()
    n_classes = len(classes)
    try:
        if n_classes == 2:
            proba = y_proba[:, 1] if np.ndim(y_proba) == 2 else y_proba
            fpr, tpr, _ = roc_curve(y_true, proba, pos_label=classes[1])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC curve",
                                      line=dict(color=PRIMARY, width=2)))
        else:
            if np.ndim(y_proba) != 2 or y_proba.shape[1] != n_classes:
                return None
            for i, cls in enumerate(classes):
                y_binary = (np.asarray(y_true) == cls).astype(int)
                fpr, tpr, _ = roc_curve(y_binary, y_proba[:, i])
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Class {cls}",
                                          line=dict(width=2)))
    except ValueError:
        return None

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                              line=dict(color=TEXT_MUTED, dash="dash")))
    fig.update_layout(
        title=title, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        template=PLOTLY_TEMPLATE, margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def plot_cluster_scatter(
    df: pd.DataFrame, x_col: str, y_col: str, cluster_col: str, title: str
) -> go.Figure:
    """Interactive scatter plot colored by cluster assignment."""
    plot_df = df.copy()
    plot_df[cluster_col] = plot_df[cluster_col].astype(str)
    fig = px.scatter(
        plot_df, x=x_col, y=y_col, color=cluster_col, template=PLOTLY_TEMPLATE,
        title=title, color_discrete_sequence=CATEGORICAL_PALETTE,
    )
    fig.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=0.5, color="white")))
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend_title="Cluster")
    return fig


def plot_cluster_size_distribution(labels: np.ndarray, title: str) -> go.Figure:
    """Bar chart of cluster sizes, including a 'Noise' bar for label -1 if present."""
    series = pd.Series(labels).astype(str).replace({"-1": "Noise"})
    counts = series.value_counts().reset_index()
    counts.columns = ["cluster", "count"]
    fig = px.bar(counts, x="cluster", y="count", template=PLOTLY_TEMPLATE, title=title,
                 color_discrete_sequence=[PRIMARY])
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> go.Figure:
    """Scatter of actual vs. predicted values with a y=x reference line."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", name="Predictions",
                              marker=dict(color=PRIMARY, size=7, opacity=0.6)))
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                              line=dict(color=TEXT_MUTED, dash="dash")))
    fig.update_layout(title=title, xaxis_title="Actual", yaxis_title="Predicted",
                       template=PLOTLY_TEMPLATE, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> go.Figure:
    """Residual plot: predicted value vs. (actual - predicted)."""
    residuals = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers",
                              marker=dict(color=PRIMARY, size=7, opacity=0.6)))
    fig.add_hline(y=0, line=dict(color=TEXT_MUTED, dash="dash"))
    fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="Residual (Actual - Predicted)",
                       template=PLOTLY_TEMPLATE, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_feature_importance(importances: dict[str, float], title: str, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of the top-N feature importances."""
    items = list(importances.items())[:top_n]
    names = [k for k, _ in items][::-1]
    values = [v for _, v in items][::-1]
    fig = px.bar(x=values, y=names, orientation="h", template=PLOTLY_TEMPLATE, title=title,
                 color_discrete_sequence=[PRIMARY])
    fig.update_layout(xaxis_title="Importance", yaxis_title="", margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_elbow_and_silhouette(
    k_values: list[int], inertias: list[float], silhouettes: list[float | None], title: str
) -> go.Figure:
    """Side-by-side elbow (inertia) and silhouette-score line charts across K."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k_values, y=inertias, mode="lines+markers", name="Inertia",
                              line=dict(color=PRIMARY)))
    valid = [(k, s) for k, s in zip(k_values, silhouettes) if s is not None]
    if valid:
        xs, ys = zip(*valid)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="Silhouette score",
                                  yaxis="y2", line=dict(color="#E67E22")))
    fig.update_layout(
        title=title, xaxis_title="Number of clusters (K)",
        yaxis=dict(title="Inertia"),
        yaxis2=dict(title="Silhouette score", overlaying="y", side="right"),
        template=PLOTLY_TEMPLATE, margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig
