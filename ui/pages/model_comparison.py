"""Model Comparison: unified view across every saved experiment this session
and from prior runs of the interactive app (loaded from results/metrics/)."""

import plotly.express as px
import streamlit as st

from src.experiment_store import load_all_results
from ui import components

TASK_METRICS = {
    "classification": ["accuracy", "precision", "recall", "f1_score", "roc_auc"],
    "boosting": ["accuracy", "precision", "recall", "f1_score", "roc_auc"],
    "regression": ["mae", "rmse", "r2"],
    "clustering": ["silhouette_score", "n_clusters", "n_noise_points"],
}


def render() -> None:
    st.header("Model Comparison")
    st.caption(
        "Aggregates results from every experiment run through this app. Classification, "
        "boosting, regression, and clustering use different, non-comparable metrics, so "
        "rankings are always computed within one task at a time."
    )

    all_results = load_all_results()
    if all_results.empty:
        components.empty_state(
            "No saved experiments yet.",
            "Run at least one experiment on the Classification, Clustering, Boosting, or "
            "Regression pages — results are saved automatically.",
        )
        return

    col1, col2 = st.columns(2)
    with col1:
        task_options = sorted(all_results["task"].unique())
        task = st.selectbox("Task", options=task_options, key="cmp_task")
    task_df = all_results[all_results["task"] == task]

    with col2:
        dataset_options = sorted(task_df["dataset_name"].unique())
        selected_datasets = st.multiselect(
            "Dataset filter", options=dataset_options, default=dataset_options, key="cmp_dataset_filter"
        )
    task_df = task_df[task_df["dataset_name"].isin(selected_datasets)]

    if task_df.empty:
        components.empty_state("No experiments match the current filters.")
        return

    available_metrics = [m for m in TASK_METRICS.get(task, []) if m in task_df.columns]
    if not available_metrics:
        st.dataframe(task_df, width="stretch")
        return

    ranking_metric = st.selectbox("Rank by", options=available_metrics, key="cmp_rank_metric")
    ascending = ranking_metric in {"mae", "rmse", "n_noise_points"}
    ranked = task_df.sort_values(ranking_metric, ascending=ascending)

    components.section_title(f"{task.title()} experiments — ranked by {ranking_metric}")
    st.dataframe(
        ranked[["experiment_id", "dataset_name", "timestamp", "model"] + available_metrics],
        width="stretch",
    )

    chart_df = ranked.copy()
    chart_df["label"] = chart_df["model"].astype(str) + " (" + chart_df["dataset_name"].astype(str) + ")"
    fig = px.bar(
        chart_df, x="label", y=ranking_metric, template="plotly_white",
        title=f"{ranking_metric} by model and dataset", color_discrete_sequence=["#8E44AD"],
    )
    fig.update_layout(xaxis_title="", margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")

    components.download_dataframe_button(
        ranked, "Download comparison CSV", f"{task}_comparison.csv", key="cmp_dl"
    )
