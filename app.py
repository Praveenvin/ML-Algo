"""ML Studio: interactive frontend for the ML Algorithms Comparison project.

Thin router only — all ML logic lives in src/, all UI logic lives in ui/.
Launch with: streamlit run app.py
"""

import streamlit as st

from ui import data_state
from ui.pages import boosting, classification, clustering, dashboard, dataset_explorer, model_comparison, regression
from ui.styles import inject_global_styles, render_header

st.set_page_config(page_title="ML Studio", page_icon="🔮", layout="wide")

PAGES = {
    "Dashboard": dashboard,
    "Dataset Explorer": dataset_explorer,
    "Classification": classification,
    "Clustering": clustering,
    "Boosting": boosting,
    "Regression": regression,
    "Model Comparison": model_comparison,
}


def main() -> None:
    data_state.init_session_state()
    inject_global_styles()

    with st.sidebar:
        st.markdown("### ML Studio")
        page_name = st.radio("Navigate", options=list(PAGES.keys()), label_visibility="collapsed")
        st.divider()
        if st.button("Clear all results", width="stretch"):
            data_state.clear_all_results()
            for key in ("classification_result", "boosting_result", "clustering_result", "regression_result"):
                st.session_state.pop(key, None)
            st.success("Cleared.")
        st.caption("ML Studio reuses the same scikit-learn pipelines as the project's CLI scripts.")

    render_header()
    PAGES[page_name].render()


if __name__ == "__main__":
    main()
