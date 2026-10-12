"""Reusable, leakage-safe preprocessing pipelines built on scikit-learn."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_classification_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Build a ColumnTransformer that scales numeric features and one-hot
    encodes categorical features. Intended to be fit on training data only,
    as the first step of a scikit-learn Pipeline.

    Args:
        numeric_features: Names of numeric columns to standard-scale.
        categorical_features: Names of categorical columns to one-hot encode.

    Returns:
        An unfitted ColumnTransformer.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), categorical_features),
        ]
    )


def build_clustering_preprocessor(features: list[str]) -> Pipeline:
    """Build a scaling pipeline for clustering features.

    Args:
        features: Numeric feature column names to scale.

    Returns:
        An unfitted scikit-learn Pipeline wrapping a StandardScaler.
    """
    return Pipeline(steps=[("scaler", StandardScaler())])


def infer_feature_types(
    df: pd.DataFrame, feature_columns: list[str]
) -> tuple[list[str], list[str]]:
    """Split feature columns into numeric and categorical based on dtype.

    Used by the interactive app, where feature columns are chosen by the
    user at runtime rather than hardcoded per dataset.

    Args:
        df: DataFrame containing the feature columns.
        feature_columns: Columns to classify.

    Returns:
        A (numeric_features, categorical_features) tuple, each preserving
        the original column order.
    """
    numeric_features = [c for c in feature_columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_features = [c for c in feature_columns if c not in numeric_features]
    return numeric_features, categorical_features


def validate_no_missing(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a ValueError if any of the given columns contain missing values.

    Args:
        df: DataFrame to validate.
        columns: Columns that must be fully populated.

    Raises:
        ValueError: If missing values are found.
    """
    missing_counts = df[columns].isna().sum()
    offenders = missing_counts[missing_counts > 0]
    if not offenders.empty:
        raise ValueError(f"Missing values detected in columns: {offenders.to_dict()}")
