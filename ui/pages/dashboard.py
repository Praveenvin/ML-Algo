"""Dashboard: at-a-glance summary of the active dataset and latest experiments."""

import pandas as pd
import streamlit as st

from ui import components, data_state
from ui.config_form import render_dataset_picker


def render() -> None:
    st.header("Dashboard")

    name, df = render_dataset_picker(key_prefix="dash")

    if df is None:
        components.empty_state(
            "No dataset selected yet.",
            "Go to <b>Dataset Explorer</b> to upload a CSV or pick one of the three sample "
            "datasets, then return here.",
        )
        _quick_start()
        return

    data_state.set_active_dataset(name)
    summary = components.dataframe_quality_summary(df)
    most_recent_task = data_state.get_most_recent_task()
    latest = data_state.get_latest_result(most_recent_task) if most_recent_task else None

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        components.metric_card("Dataset", name.replace(" (sample)", ""))
    with col2:
        components.metric_card("Rows", f"{summary['n_rows']:,}")
    with col3:
        components.metric_card("Features", f"{summary['n_columns']}")
    with col4:
        components.metric_card("Missing values", f"{summary['n_missing']}")
    with col5:
        components.metric_card("Last ML task", (most_recent_task or "None").title())
    with col6:
        best_model = latest.get("best_model_name") if latest else None
        components.metric_card("Best model", best_model or "—")

    components.section_title("Dataset overview")
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        default_col = numeric_cols[0]
        col = st.selectbox("Preview column distribution", options=df.columns.tolist(),
                            index=df.columns.tolist().index(default_col), key="dash_preview_col")
        st.plotly_chart(components.plot_distribution(df, col), width="stretch")

    if latest is not None and "metrics_df" in latest:
        components.section_title(f"Latest {most_recent_task} comparison — dataset: {latest['dataset_name']}")
        metrics_df = latest["metrics_df"]
        plottable = [c for c in ["accuracy", "precision", "recall", "f1_score", "r2"] if c in metrics_df.columns]
        if plottable:
            st.plotly_chart(
                components.plot_metric_comparison(metrics_df, plottable, "Model comparison"),
                width="stretch",
            )
        st.dataframe(metrics_df, width="stretch")
    else:
        components.empty_state(
            "No experiments completed yet.",
            "Head to Classification, Clustering, Boosting, or Regression and click "
            "<b>Run Experiment</b> to see results here.",
        )

    _quick_start()


def _quick_start() -> None:
    with st.expander("Quick start", expanded=False):
        st.markdown(
            """
            1. Open **Dataset Explorer** and either upload a CSV or pick one of the three
               bundled sample datasets (`Social_Network_Ads.csv`, `Mall_Customers.csv`,
               `insurance_pre.csv`).
            2. Go to **Classification**, **Clustering**, **Boosting**, or **Regression**
               depending on your task.
            3. Choose the target column (if supervised) and feature columns, pick one or
               more algorithms, and adjust the test split / random seed if needed.
            4. Click **Run Experiment** to train and evaluate the selected models.
            5. Review metrics and charts, download results as CSV, and compare everything
               across experiments on the **Model Comparison** page.
            """
        )
