"""Boosting page: configure, run, and inspect boosting-ensemble experiments."""

import streamlit as st

from src.boosting import compare_with_classification, get_models, run_boosting_experiment
from src.experiment_store import save_experiment
from ui import components, data_state
from ui.config_form import (
    render_dataset_picker,
    render_feature_target_selector,
    render_model_selector,
    render_run_button,
    render_split_and_seed_controls,
)

RESULT_KEY = "boosting_result"


def render() -> None:
    st.header("Boosting")
    st.caption("Train and compare boosting ensembles, and benchmark them against the "
               "standard classifiers from the Classification page.")

    name, df = render_dataset_picker(key_prefix="boost")
    if df is None:
        components.empty_state("No dataset selected.", "Pick one in Dataset Explorer first.")
        return

    feature_columns, target_column = render_feature_target_selector(df, key_prefix="boost")
    model_names = render_model_selector(list(get_models().keys()), key_prefix="boost")
    test_size, seed = render_split_and_seed_controls(key_prefix="boost")

    if render_run_button("Run Experiment", key_prefix="boost"):
        try:
            with st.spinner("Training boosting models..."):
                result = run_boosting_experiment(
                    df, feature_columns, target_column,
                    model_names=model_names, test_size=test_size, random_state=seed,
                )
        except ValueError as exc:
            st.error(str(exc))
        else:
            experiment_id = save_experiment(
                task="boosting", dataset_name=name, metrics_df=result["metrics_df"],
                best_model=result["best_model_name"], selection_metric=result["selection_metric"],
                target_column=target_column,
            )
            result["experiment_id"] = experiment_id
            result["target_column"] = target_column
            st.session_state[RESULT_KEY] = {"dataset_name": name, **result}
            data_state.record_latest_result("boosting", name, result)
            st.success(f"Experiment complete. Best model: {result['best_model_name']}")

    stored = st.session_state.get(RESULT_KEY)
    if stored is None or stored["dataset_name"] != name:
        if stored is not None:
            components.empty_state("Switched dataset.", "Run a new experiment to see results for this dataset.")
        return

    _render_results(stored)


def _render_results(result: dict) -> None:
    metrics_df = result["metrics_df"]

    components.section_title("Boosting model comparison (test-set metrics)")
    metric_choices = [c for c in ["accuracy", "precision", "recall", "f1_score", "roc_auc"] if c in metrics_df.columns]
    st.plotly_chart(
        components.plot_metric_comparison(metrics_df, metric_choices, "Boosting model comparison"),
        width="stretch",
    )
    st.caption(
        f"Best model selected by: {result['selection_metric']}. The table below separates "
        f"test-set metrics (accuracy, precision, recall, f1_score, roc_auc) from the "
        f"cross-validated selection score (cv_score_mean) used only to pick the best model."
    )
    st.dataframe(metrics_df, width="stretch")
    components.download_dataframe_button(
        metrics_df, "Download boosting metrics CSV", "boosting_metrics.csv", key="boost_dl_metrics"
    )

    components.section_title("Confusion matrices")
    classes = [str(c) for c in result["classes"]]
    model_names = list(result["confusion_matrices"].keys())
    cols = st.columns(min(3, len(model_names)))
    for i, model_name in enumerate(model_names):
        with cols[i % len(cols)]:
            st.plotly_chart(
                components.plot_confusion_matrix(result["confusion_matrices"][model_name], classes, model_name),
                width="stretch",
            )

    components.section_title("Feature importance")
    model_with_importance = [m for m, imp in result["feature_importances"].items() if imp is not None]
    if model_with_importance:
        focus_model = st.selectbox("Model", options=model_with_importance, key="boost_focus_model")
        st.plotly_chart(
            components.plot_feature_importance(
                result["feature_importances"][focus_model], f"Feature importance — {focus_model}"
            ),
            width="stretch",
        )
    else:
        st.info("None of the selected boosting models expose feature importances.")

    classification_result = data_state.get_latest_result("classification")
    if (
        classification_result is not None
        and classification_result["dataset_name"] == result["dataset_name"]
        and classification_result.get("target_column") == result.get("target_column")
    ):
        components.section_title("Boosting vs. standard classifiers")
        combined = compare_with_classification(classification_result["metrics_df"], metrics_df)
        st.dataframe(combined, width="stretch")
        components.download_dataframe_button(
            combined, "Download combined comparison CSV", "boosting_vs_classification.csv",
            key="boost_dl_combined",
        )
    else:
        st.info(
            "Run a Classification experiment on the same dataset and target column to unlock "
            "a side-by-side comparison with boosting."
        )
