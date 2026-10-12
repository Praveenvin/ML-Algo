"""Regression page: configure, run, and inspect regression experiments."""

import streamlit as st

from src.regression import get_models, run_regression_experiment
from src.experiment_store import save_experiment
from ui import components, data_state
from ui.config_form import (
    render_dataset_picker,
    render_feature_target_selector,
    render_model_selector,
    render_run_button,
    render_split_and_seed_controls,
)

RESULT_KEY = "regression_result"


def render() -> None:
    st.header("Regression")
    st.caption("Predict a numeric target column from the selected features.")

    name, df = render_dataset_picker(key_prefix="reg")
    if df is None:
        components.empty_state("No dataset selected.", "Pick one in Dataset Explorer first.")
        return

    feature_columns, target_column = render_feature_target_selector(
        df, key_prefix="reg", numeric_target_only=True,
    )
    model_names = render_model_selector(list(get_models().keys()), key_prefix="reg")
    test_size, seed = render_split_and_seed_controls(key_prefix="reg")

    if render_run_button("Run Experiment", key_prefix="reg"):
        try:
            with st.spinner("Training regressors..."):
                result = run_regression_experiment(
                    df, feature_columns, target_column,
                    model_names=model_names, test_size=test_size, random_state=seed,
                )
        except ValueError as exc:
            st.error(str(exc))
        else:
            experiment_id = save_experiment(
                task="regression", dataset_name=name, metrics_df=result["metrics_df"],
                best_model=result["best_model_name"], selection_metric=result["selection_metric"],
                target_column=target_column,
            )
            result["experiment_id"] = experiment_id
            result["target_column"] = target_column
            st.session_state[RESULT_KEY] = {"dataset_name": name, **result}
            data_state.record_latest_result("regression", name, result)
            st.success(f"Experiment complete. Best model: {result['best_model_name']}")

    stored = st.session_state.get(RESULT_KEY)
    if stored is None or stored["dataset_name"] != name:
        if stored is not None:
            components.empty_state("Switched dataset.", "Run a new experiment to see results for this dataset.")
        return

    _render_results(stored)


def _render_results(result: dict) -> None:
    metrics_df = result["metrics_df"]

    components.section_title("Regression model comparison")
    st.plotly_chart(
        components.plot_regression_comparison(metrics_df, "Regression model comparison"),
        width="stretch",
    )
    st.caption(f"Best model selected by: {result['selection_metric']}")
    st.dataframe(metrics_df, width="stretch")
    components.download_dataframe_button(
        metrics_df, "Download regression metrics CSV", "regression_metrics.csv", key="reg_dl_metrics"
    )
    components.download_dataframe_button(
        result["predictions_df"], "Download predictions CSV", "regression_predictions.csv",
        key="reg_dl_preds",
    )

    components.section_title("Actual vs. predicted / residuals")
    model_names = list(result["plot_data"].keys())
    focus_model = st.selectbox("Model", options=model_names, index=model_names.index(result["best_model_name"]),
                                key="reg_focus_model")
    y_true, y_pred = result["plot_data"][focus_model]

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            components.plot_actual_vs_predicted(y_true, y_pred, f"Actual vs. Predicted — {focus_model}"),
            width="stretch",
        )
    with col2:
        st.plotly_chart(
            components.plot_residuals(y_true, y_pred, f"Residuals — {focus_model}"),
            width="stretch",
        )
