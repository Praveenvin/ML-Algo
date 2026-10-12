"""Tests for preprocessing pipelines and validation helpers."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    build_classification_preprocessor,
    build_clustering_preprocessor,
    infer_feature_types,
    validate_no_missing,
)


def test_validate_no_missing_passes_on_clean_data() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    validate_no_missing(df, ["a", "b"])  # should not raise


def test_validate_no_missing_raises_on_missing_values() -> None:
    df = pd.DataFrame({"a": [1, None, 3]})
    with pytest.raises(ValueError):
        validate_no_missing(df, ["a"])


def test_classification_preprocessor_fits_on_train_only() -> None:
    df = pd.DataFrame(
        {
            "Age": [20, 30, 40, 50],
            "EstimatedSalary": [20000, 30000, 40000, 50000],
            "Gender": ["Male", "Female", "Male", "Female"],
        }
    )
    preprocessor = build_classification_preprocessor(["Age", "EstimatedSalary"], ["Gender"])
    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == len(df)
    # Age/EstimatedSalary scaled (mean ~0) + one-hot encoded Gender column(s)
    assert transformed.shape[1] >= 3


def test_clustering_preprocessor_scales_features() -> None:
    X = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
    scaler = build_clustering_preprocessor(["a", "b"])
    X_scaled = scaler.fit_transform(X)
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)


def test_infer_feature_types_splits_numeric_and_categorical() -> None:
    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "income": [1000.0, 2000.0, 3000.0],
            "gender": ["M", "F", "M"],
            "city": ["A", "B", "C"],
        }
    )
    numeric, categorical = infer_feature_types(df, ["age", "income", "gender", "city"])
    assert numeric == ["age", "income"]
    assert categorical == ["gender", "city"]


def test_infer_feature_types_preserves_order_and_subset() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [1.5, 2.5]})
    numeric, categorical = infer_feature_types(df, ["b", "c", "a"])
    assert numeric == ["c", "a"]
    assert categorical == ["b"]
