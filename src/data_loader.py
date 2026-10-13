"""Dataset loading and schema validation utilities."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def resolve_dataset_path(path: Path | str) -> Path:
    """Resolve a dataset path relative to the project root, not the current working directory."""
    p = Path(path)
    if not p.is_absolute():
        # Check relative to project root directly (e.g. "data/Social_Network_Ads.csv")
        candidate_root = PROJECT_ROOT / p
        if candidate_root.exists():
            return candidate_root
        # Check relative to DATA_DIR (e.g. "Social_Network_Ads.csv")
        candidate_data = DATA_DIR / p
        if candidate_data.exists():
            return candidate_data
        # Default candidate if neither exists yet
        p = candidate_root if "data" in p.parts else candidate_data

    # Case-insensitive resolution for Linux/Streamlit Cloud
    if not p.exists() and p.parent.is_dir():
        target_name = p.name.lower()
        for file in p.parent.iterdir():
            if file.is_file() and file.name.lower() == target_name:
                return file

    return p


def load_csv(path: Path | str, required_columns: list[str] | None = None) -> pd.DataFrame:
    """Load a CSV file and optionally validate that required columns are present.

    Args:
        path: Path to the CSV file (absolute, or relative to project root or data/).
        required_columns: Columns that must exist in the loaded DataFrame.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    resolved_path = resolve_dataset_path(path)

    if not resolved_path.exists():
        available_files: list[str] = []
        if DATA_DIR.is_dir():
            available_files = sorted([f.name for f in DATA_DIR.glob("*.csv")])
        raise FileNotFoundError(
            f"Dataset not found at '{resolved_path}'. "
            f"Available dataset files in '{DATA_DIR}': {available_files}."
        )

    df = pd.read_csv(resolved_path)

    if required_columns is not None:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"Dataset at '{resolved_path}' is missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

    return df


def load_social_network_ads(path: Path | str | None = None) -> pd.DataFrame:
    """Load the Social_Network_Ads dataset used for classification and boosting."""
    if path is None:
        path = DATA_DIR / "Social_Network_Ads.csv"
    required = ["User ID", "Gender", "Age", "EstimatedSalary", "Purchased"]
    df = load_csv(path, required)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def load_mall_customers(path: Path | str | None = None) -> pd.DataFrame:
    """Load the Mall_Customers dataset used for clustering."""
    if path is None:
        path = DATA_DIR / "Mall_Customers.csv"
    required = ["CustomerID", "Genre", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
    df = load_csv(path, required)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def load_insurance(path: Path | str | None = None) -> pd.DataFrame:
    """Load the insurance_pre dataset used for the optional regression module."""
    if path is None:
        path = DATA_DIR / "insurance_pre.csv"
    required = ["age", "sex", "bmi", "children", "smoker", "charges"]
    df = load_csv(path, required)
    df = df.drop_duplicates().reset_index(drop=True)
    return df
