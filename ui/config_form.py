"""Shared experiment-configuration widgets used by every algorithm page.

Each `render_*` function returns plain Python values (never trains a model
itself), so pages stay in control of when training actually runs.
"""

import pandas as pd
import streamlit as st

from ui import data_state


def render_dataset_picker(key_prefix: str) -> tuple[str | None, pd.DataFrame | None]:
    """Render a dataset selector and return (dataset_name, dataframe) or (None, None).

    Defaults to the globally active dataset (set in Dataset Explorer) when available.
    """
    names = data_state.available_dataset_names()
    if not names:
        return None, None

    active = data_state.get_active_dataset_name()
    default_index = names.index(active) if active in names else 0

    name = st.selectbox("Dataset", options=names, index=default_index, key=f"{key_prefix}_dataset")
    try:
        df = data_state.get_dataset(name)
    except KeyError:
        return name, None
    except FileNotFoundError as exc:
        st.error(str(exc))
        return name, None
    return name, df


def render_feature_target_selector(
    df: pd.DataFrame,
    key_prefix: str,
    require_target: bool = True,
    numeric_target_only: bool = False,
    numeric_features_only: bool = False,
) -> tuple[list[str], str | None]:
    """Render target + feature-column multiselects inferred from the dataframe.

    Args:
        df: Active dataset.
        key_prefix: Unique widget key prefix for this page.
        require_target: Whether to show a target selector (False for clustering).
        numeric_target_only: Restrict the target selector to numeric columns (regression).
        numeric_features_only: Restrict the feature selector to numeric columns (clustering).

    Returns:
        (feature_columns, target_column). target_column is None when
        require_target is False.
    """
    all_columns = list(df.columns)
    numeric_columns = [c for c in all_columns if pd.api.types.is_numeric_dtype(df[c])]

    target_column = None
    if require_target:
        target_candidates = numeric_columns if numeric_target_only else all_columns
        if not target_candidates:
            st.error("No suitable target columns found in this dataset.")
            return [], None
        default_target_index = len(target_candidates) - 1
        target_column = st.selectbox(
            "Target column", options=target_candidates, index=default_target_index,
            key=f"{key_prefix}_target",
        )

    feature_candidates = numeric_columns if numeric_features_only else all_columns
    feature_candidates = [c for c in feature_candidates if c != target_column]
    # Drop obvious ID-like columns from the default selection (e.g. CustomerID, User ID)
    default_features = [
        c for c in feature_candidates
        if "id" not in c.lower().replace(" ", "")
    ]
    if not default_features:
        default_features = feature_candidates

    feature_columns = st.multiselect(
        "Feature columns", options=feature_candidates, default=default_features,
        key=f"{key_prefix}_features",
    )
    return feature_columns, target_column


def render_model_selector(model_options: list[str], key_prefix: str) -> list[str]:
    """Render a checkbox-style multiselect of algorithms, defaulting to all."""
    return st.multiselect(
        "Algorithms", options=model_options, default=model_options, key=f"{key_prefix}_models"
    )


def render_split_and_seed_controls(key_prefix: str) -> tuple[float, int]:
    """Render test-split percentage and random-seed inputs. Returns (test_size, seed)."""
    col1, col2 = st.columns(2)
    with col1:
        test_pct = st.slider("Test split (%)", min_value=10, max_value=40, value=20, step=5,
                              key=f"{key_prefix}_test_pct")
    with col2:
        seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1,
                                key=f"{key_prefix}_seed")
    return test_pct / 100.0, int(seed)


def render_run_button(label: str, key_prefix: str) -> bool:
    """Render the prominent 'Run Experiment' button."""
    return st.button(label, type="primary", width="stretch", key=f"{key_prefix}_run")
