"""Session-state-backed dataset registry shared by every ML Studio page.

Datasets (sample or uploaded) live only in `st.session_state` — never written
back to `data/`, so the original CSVs on disk are never overwritten by the app.
"""

from typing import Any

import pandas as pd
import streamlit as st

from src import config
from src.data_loader import load_insurance, load_mall_customers, load_social_network_ads

SAMPLE_DATASETS = {
    "Social_Network_Ads.csv (sample)": (load_social_network_ads, config.SOCIAL_NETWORK_ADS_PATH),
    "Mall_Customers.csv (sample)": (load_mall_customers, config.MALL_CUSTOMERS_PATH),
    "insurance_pre.csv (sample)": (load_insurance, config.INSURANCE_PATH),
}


def init_session_state() -> None:
    """Initialize all ML Studio session-state keys on first run."""
    st.session_state.setdefault("datasets", {})
    st.session_state.setdefault("active_dataset_name", None)
    st.session_state.setdefault("latest_results", {})  # task -> result dict, for the Dashboard
    st.session_state.setdefault("last_task", None)


def available_dataset_names() -> list[str]:
    """List sample dataset names plus any datasets uploaded this session."""
    uploaded = [n for n in st.session_state["datasets"] if n not in SAMPLE_DATASETS]
    return list(SAMPLE_DATASETS.keys()) + uploaded


def get_dataset(name: str) -> pd.DataFrame:
    """Load (and cache in session state) the dataset with the given name.

    Sample datasets are loaded from disk via the existing src.data_loader
    functions the first time they're requested. Uploaded datasets must
    already be registered via `register_uploaded_dataset`.

    Raises:
        FileNotFoundError: If a sample dataset's underlying CSV is missing.
        KeyError: If `name` is neither a known sample nor a registered upload.
    """
    if name in st.session_state["datasets"]:
        return st.session_state["datasets"][name]

    if name in SAMPLE_DATASETS:
        loader, path = SAMPLE_DATASETS[name]
        df = loader(path)
        st.session_state["datasets"][name] = df
        return df

    raise KeyError(f"Dataset '{name}' is not loaded. Upload it first in Dataset Explorer.")


def register_uploaded_dataset(filename: str, df: pd.DataFrame) -> str:
    """Register an uploaded DataFrame under a unique session-state key.

    Returns:
        The key the dataset was registered under (the filename, possibly
        de-duplicated if a dataset with the same name was already uploaded).
    """
    name = filename
    suffix = 2
    while name in st.session_state["datasets"] and name not in SAMPLE_DATASETS:
        existing = st.session_state["datasets"].get(name)
        if existing is not None and existing.equals(df):
            break
        name = f"{filename} ({suffix})"
        suffix += 1

    st.session_state["datasets"][name] = df
    return name


def set_active_dataset(name: str) -> None:
    st.session_state["active_dataset_name"] = name


def get_active_dataset_name() -> str | None:
    return st.session_state.get("active_dataset_name")


def record_latest_result(task: str, dataset_name: str, result: dict[str, Any]) -> None:
    """Remember the most recent result per task, for the Dashboard summary."""
    st.session_state["latest_results"][task] = {"dataset_name": dataset_name, **result}
    st.session_state["last_task"] = task


def get_latest_result(task: str) -> dict[str, Any] | None:
    return st.session_state["latest_results"].get(task)


def get_most_recent_task() -> str | None:
    """Return the task name of whichever experiment was run most recently, if any."""
    return st.session_state.get("last_task")


def clear_all_results() -> None:
    """Reset all stored experiment results (used by the 'Clear results' action)."""
    st.session_state["latest_results"] = {}
    st.session_state["last_task"] = None
