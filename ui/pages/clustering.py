"""Clustering page: configure, run, and inspect clustering experiments."""

import pandas as pd
import streamlit as st

from src.clustering import ALGORITHM_FACTORY_NAMES, run_clustering_experiment
from src.experiment_store import save_experiment
from ui import components, data_state
from ui.config_form import render_dataset_picker, render_feature_target_selector, render_run_button

RESULT_KEY = "clustering_result"


def render() -> None:
    st.header("Clustering")
    st.caption("Unsupervised segmentation — no target column required.")

    name, df = render_dataset_picker(key_prefix="clust")
    if df is None:
        components.empty_state("No dataset selected.", "Pick one in Dataset Explorer first.")
        return

    feature_columns, _ = render_feature_target_selector(
        df, key_prefix="clust", require_target=False, numeric_features_only=True,
    )
    if len(feature_columns) < 2:
        st.warning("Select at least 2 numeric feature columns to run clustering.")

    algorithms = st.multiselect(
        "Algorithms", options=list(ALGORITHM_FACTORY_NAMES), default=list(ALGORITHM_FACTORY_NAMES),
        key="clust_algorithms",
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kmeans_k = st.number_input("K-Means: number of clusters", min_value=2, max_value=15, value=5,
                                    key="clust_kmeans_k")
    with col2:
        agglomerative_k = st.number_input("Agglomerative: number of clusters", min_value=2, max_value=15,
                                           value=5, key="clust_agg_k")
    with col3:
        dbscan_eps_mode = st.selectbox("DBSCAN eps", options=["Auto-estimate", "Manual"], key="clust_eps_mode")
        dbscan_eps = None
        if dbscan_eps_mode == "Manual":
            dbscan_eps = st.number_input("DBSCAN eps value", min_value=0.01, value=0.5, step=0.05,
                                          key="clust_eps_value")
    with col4:
        dbscan_min_samples = st.number_input("DBSCAN min samples", min_value=2, max_value=50, value=5,
                                              key="clust_min_samples")

    if render_run_button("Run Experiment", key_prefix="clust"):
        try:
            with st.spinner("Running clustering algorithms..."):
                result = run_clustering_experiment(
                    df, feature_columns, algorithms=algorithms,
                    kmeans_k=int(kmeans_k), agglomerative_k=int(agglomerative_k),
                    dbscan_eps=dbscan_eps, dbscan_min_samples=int(dbscan_min_samples),
                )
        except ValueError as exc:
            st.error(str(exc))
        else:
            experiment_id = save_experiment(
                task="clustering", dataset_name=name, metrics_df=result["metrics_df"],
            )
            result["experiment_id"] = experiment_id
            st.session_state[RESULT_KEY] = {
                "dataset_name": name, "feature_columns": feature_columns, **result,
            }
            data_state.record_latest_result("clustering", name, result)
            st.success("Clustering complete.")

    stored = st.session_state.get(RESULT_KEY)
    if stored is None or stored["dataset_name"] != name:
        if stored is not None:
            components.empty_state("Switched dataset.", "Run a new experiment to see results for this dataset.")
        return

    _render_results(stored)


def _render_results(result: dict) -> None:
    metrics_df = result["metrics_df"]
    numeric_features = result["numeric_features"]
    assignments_df = result["assignments_df"]

    components.section_title("Clustering metrics")
    st.dataframe(metrics_df, width="stretch")
    components.download_dataframe_button(
        metrics_df, "Download clustering metrics CSV", "clustering_metrics.csv", key="clust_dl_metrics"
    )
    components.download_dataframe_button(
        assignments_df, "Download cluster assignments CSV", "cluster_assignments.csv", key="clust_dl_assign"
    )

    cluster_columns = [c for c in assignments_df.columns if c.endswith("_cluster")]
    if not cluster_columns:
        return

    components.section_title("Cluster scatter plot")
    use_pca = result.get("pca_df") is not None
    if use_pca and len(numeric_features) > 2:
        st.caption(
            f"{len(numeric_features)} features selected — showing a 2D PCA projection "
            f"(explains {sum(result['pca_explained_variance_ratio']) * 100:.1f}% of variance)."
        )
        x_col, y_col = "PC1", "PC2"
        plot_source = pd.concat([result["pca_df"], assignments_df[cluster_columns].reset_index(drop=True)], axis=1)
    else:
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("X axis", options=numeric_features, index=0, key="clust_x_axis")
        with col2:
            default_y = 1 if len(numeric_features) > 1 else 0
            y_col = st.selectbox("Y axis", options=numeric_features, index=default_y, key="clust_y_axis")
        plot_source = assignments_df

    algo_tabs = st.tabs(cluster_columns)
    for tab, cluster_col in zip(algo_tabs, cluster_columns):
        with tab:
            algo_name = cluster_col.replace("_cluster", "")
            st.plotly_chart(
                components.plot_cluster_scatter(plot_source, x_col, y_col, cluster_col, f"{algo_name} clusters"),
                width="stretch",
            )
            st.plotly_chart(
                components.plot_cluster_size_distribution(
                    assignments_df[cluster_col].values, f"{algo_name} cluster sizes"
                ),
                width="stretch",
            )
            if (assignments_df[cluster_col] == -1).any():
                st.caption(f"{int((assignments_df[cluster_col] == -1).sum())} point(s) labeled as noise (-1).")

    components.section_title("Cluster profiles (K-Means)")
    if "KMeans_cluster" in assignments_df.columns:
        profile_cols = numeric_features
        profiles = assignments_df.groupby("KMeans_cluster")[profile_cols].mean().round(2)
        profiles["count"] = assignments_df.groupby("KMeans_cluster").size()
        st.dataframe(profiles, width="stretch")
