"""Dataset Explorer: upload a CSV or pick a sample dataset, and inspect it."""

import pandas as pd
import streamlit as st

from ui import components, data_state


def render() -> None:
    st.header("Dataset Explorer")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as exc:  # malformed CSV, wrong encoding, etc.
            st.error(f"Could not read '{uploaded_file.name}' as a CSV: {exc}")
            df = None
        if df is not None:
            if df.empty:
                st.error(f"'{uploaded_file.name}' has no rows.")
            else:
                registered_name = data_state.register_uploaded_dataset(uploaded_file.name, df)
                data_state.set_active_dataset(registered_name)
                st.success(f"Loaded '{registered_name}' ({len(df):,} rows, {df.shape[1]} columns).")

    st.markdown("**Or pick a dataset already available:**")
    names = data_state.available_dataset_names()
    active = data_state.get_active_dataset_name()
    default_index = names.index(active) if active in names else 0
    selected_name = st.selectbox("Active dataset", options=names, index=default_index, key="explorer_dataset")
    data_state.set_active_dataset(selected_name)

    try:
        df = data_state.get_dataset(selected_name)
    except (KeyError, FileNotFoundError) as exc:
        st.error(str(exc))
        return

    summary = components.dataframe_quality_summary(df)

    components.section_title(f"Overview — {selected_name}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        components.metric_card("Rows", f"{summary['n_rows']:,}")
    with col2:
        components.metric_card("Columns", f"{summary['n_columns']}")
    with col3:
        components.metric_card("Missing values", f"{summary['n_missing']}")
    with col4:
        components.metric_card("Duplicate rows", f"{summary['n_duplicates']}")

    render_quality_warnings_and_dtypes(df, summary)

    components.section_title("Data preview")
    search = st.text_input("Search (matches any cell, case-insensitive)", key="explorer_search")
    preview_df = df
    if search:
        mask = df.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
        preview_df = df[mask]
        st.caption(f"{len(preview_df)} row(s) match '{search}'.")
    st.dataframe(preview_df, width="stretch", height=320)

    components.section_title("Summary statistics")
    st.dataframe(df.describe(include="all").transpose(), width="stretch")

    components.section_title("Column distribution")
    column = st.selectbox("Choose a column to visualize", options=df.columns.tolist(), key="explorer_dist_col")
    st.plotly_chart(components.plot_distribution(df, column), width="stretch")


def render_quality_warnings_and_dtypes(df: pd.DataFrame, summary: dict) -> None:
    components.section_title("Data types & quality")
    dtype_df = pd.DataFrame({
        "column": list(summary["dtypes"].keys()),
        "dtype": list(summary["dtypes"].values()),
        "missing": [int(df[c].isna().sum()) for c in summary["dtypes"]],
    })
    st.dataframe(dtype_df, width="stretch", height=min(300, 40 + 35 * len(dtype_df)))
    components.render_quality_warnings(summary)
