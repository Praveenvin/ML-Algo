"""Dataset loading and schema validation utilities."""

from pathlib import Path

import pandas as pd


def load_csv(path: Path, required_columns: list[str] | None = None) -> pd.DataFrame:
    """Load a CSV file and optionally validate that required columns are present.

    Args:
        path: Path to the CSV file.
        required_columns: Columns that must exist in the loaded DataFrame.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Ensure the file is present in the data/ directory."
        )

    df = pd.read_csv(path)

    if required_columns is not None:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"Dataset at '{path}' is missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

    return df


def load_social_network_ads(path: Path) -> pd.DataFrame:
    """Load the Social_Network_Ads dataset used for classification and boosting."""
    required = ["User ID", "Gender", "Age", "EstimatedSalary", "Purchased"]
    df = load_csv(path, required)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def load_mall_customers(path: Path) -> pd.DataFrame:
    """Load the Mall_Customers dataset used for clustering."""
    required = ["CustomerID", "Genre", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
    df = load_csv(path, required)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def load_insurance(path: Path) -> pd.DataFrame:
    """Load the insurance_pre dataset used for the optional regression module."""
    required = ["age", "sex", "bmi", "children", "smoker", "charges"]
    df = load_csv(path, required)
    df = df.drop_duplicates().reset_index(drop=True)
    return df
