"""Classification page: configure, run, and inspect classifier experiments."""

import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report

from src.classification import get_models, run_classification_experiment
from src.experiment_store import save_experiment
from ui import components, data_state
from ui.config_form import (
    render_dataset_picker,
    render_feature_target_selector,
    render_model_selector,
    render_run_button,
    render_split_and_seed_controls,
)

RESULT_KEY = "classification_result"


def render() -> None:
    st.header("Classification")
    st.caption("Train and compare classifiers on a labeled target column.")

    name, df = render_dataset_picker(key_prefix="clf")
    if df is None:
        components.empty_state("No dataset selected.", "Pick one in Dataset Explorer first.")
        return

    feature_columns, target_column = render_feature_target_selector(df, key_prefix="clf")
    model_names = render_model_selector(list(get_models().keys()), key_prefix="clf")
    test_size, seed = render_split_and_seed_controls(key_prefix="clf")

    if render_run_button("Run Experiment", key_prefix="clf"):
        try:
            with st.spinner("Training classifiers..."):
                result = run_classification_experiment(
                    df, feature_columns, target_column,
                    model_names=model_names, test_size=test_size, random_state=seed,
                )
        except ValueError as exc:
            st.error(str(exc))
        else:
            experiment_id = save_experiment(
                task="classification", dataset_name=name, metrics_df=result["metrics_df"],
                best_model=result["best_model_name"], selection_metric=result["selection_metric"],
                target_column=target_column,
            )
            result["experiment_id"] = experiment_id
            result["target_column"] = target_column
            st.session_state[RESULT_KEY] = {"dataset_name": name, **result}
            data_state.record_latest_result("classification", name, result)
            st.success(f"Experiment complete. Best model: {result['best_model_name']}")

    stored = st.session_state.get(RESULT_KEY)
    if stored is None or stored["dataset_name"] != name:
        if stored is not None:
            components.empty_state("Switched dataset.", "Run a new experiment to see results for this dataset.")
        return

    _render_results(stored)


def _render_results(result: dict) -> None:
    metrics_df = result["metrics_df"]

    components.section_title("Model comparison")
    metric_choices = [c for c in ["accuracy", "precision", "recall", "f1_score", "roc_auc"] if c in metrics_df.columns]
    selected_metrics = st.multiselect(
        "Metrics to compare", options=metric_choices,
        default=[m for m in ["accuracy", "precision", "recall", "f1_score"] if m in metric_choices],
        key="clf_metric_choices",
    )
    if selected_metrics:
        st.plotly_chart(
            components.plot_metric_comparison(metrics_df, selected_metrics, "Classification model comparison"),
            width="stretch",
        )
    st.caption(f"Best model selected by: {result['selection_metric']}")
    st.dataframe(metrics_df, width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        components.download_dataframe_button(
            metrics_df, "Download metrics CSV", "classification_metrics.csv", key="clf_dl_metrics"
        )
    with col_b:
        components.download_dataframe_button(
            result["predictions_df"], "Download predictions CSV", "classification_predictions.csv",
            key="clf_dl_preds",
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

    components.section_title("Classification report & ROC curve")
    focus_model = st.selectbox("Model", options=model_names, index=model_names.index(result["best_model_name"]),
                                key="clf_focus_model")
    y_true = result["y_test"]
    y_pred = result["predictions_df"][f"{focus_model}_pred"].values
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df, width="stretch")

    y_proba = result["probabilities"].get(focus_model)
    if y_proba is not None:
        roc_fig = components.plot_roc_curve(
            y_true, y_proba, result["classes"], f"ROC curve — {focus_model}"
        )
        if roc_fig is not None:
            st.plotly_chart(roc_fig, width="stretch")
        else:
            st.info("ROC-AUC is not available for this model/target combination.")
    else:
        st.info(f"{focus_model} does not expose predicted probabilities, so no ROC curve is shown.")
