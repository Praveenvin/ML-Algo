"""Tests for dataset loading and schema validation."""

from pathlib import Path

import pandas as pd
import pytest

from src import config
from src.data_loader import (
    load_csv,
    load_insurance,
    load_mall_customers,
    load_social_network_ads,
)


def test_load_csv_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "does_not_exist.csv")


def test_load_csv_missing_required_column_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        load_csv(csv_path, required_columns=["a", "b"])


def test_load_social_network_ads_schema() -> None:
    df = load_social_network_ads(config.SOCIAL_NETWORK_ADS_PATH)
    assert {"User ID", "Gender", "Age", "EstimatedSalary", "Purchased"}.issubset(df.columns)
    assert df["Purchased"].isin([0, 1]).all()
    assert not df.isna().any().any()


def test_load_mall_customers_schema() -> None:
    df = load_mall_customers(config.MALL_CUSTOMERS_PATH)
    assert {"CustomerID", "Genre", "Annual Income (k$)", "Spending Score (1-100)"}.issubset(df.columns)
    assert not df.isna().any().any()


def test_load_insurance_schema_and_dedup() -> None:
    df = load_insurance(config.INSURANCE_PATH)
    assert {"age", "sex", "bmi", "children", "smoker", "charges"}.issubset(df.columns)
    assert df.duplicated().sum() == 0
