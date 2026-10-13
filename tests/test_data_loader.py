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


def test_load_csv_missing_file_helpful_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "non_existent_dataset.csv"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_csv(missing_path)
    error_msg = str(exc_info.value)
    assert str(missing_path) in error_msg
    assert "Available dataset files" in error_msg


def test_load_datasets_default_paths() -> None:
    df_ads = load_social_network_ads()
    assert len(df_ads) > 0
    df_mall = load_mall_customers()
    assert len(df_mall) > 0
    df_ins = load_insurance()
    assert len(df_ins) > 0


def test_load_datasets_relative_paths() -> None:
    df_rel1 = load_social_network_ads("data/Social_Network_Ads.csv")
    assert len(df_rel1) > 0
    df_rel2 = load_social_network_ads("Social_Network_Ads.csv")
    assert len(df_rel2) > 0


def test_load_datasets_from_different_working_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    df_ads = load_social_network_ads("data/Social_Network_Ads.csv")
    assert len(df_ads) > 0
    df_default = load_social_network_ads()
    assert len(df_default) > 0
