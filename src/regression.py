"""Optional Module D: Regression on insurance_pre.csv."""

from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src import config
from src.data_loader import load_insurance
from src.evaluation import evaluate_regressor
from src.preprocessing import (
    build_classification_preprocessor,
    infer_feature_types,
    validate_no_missing,
)

NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker"]

MODEL_SELECTION_METRIC = "r2"


def get_models() -> dict[str, Any]:
    """Return the regressors evaluated in this module."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=config.RANDOM_SEED),
        "Random Forest": RandomForestRegressor(random_state=config.RANDOM_SEED, n_estimators=200),
        "Gradient Boosting": GradientBoostingRegressor(random_state=config.RANDOM_SEED),
    }


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate and split the insurance dataframe into X, y."""
    validate_no_missing(df, NUMERIC_FEATURES + CATEGORICAL_FEATURES + [config.REGRESSION_TARGET])
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[config.REGRESSION_TARGET]
    return X, y


def run_regression() -> dict[str, Any]:
    """Train and evaluate all regression models.

    Returns:
        Dictionary with metrics DataFrame, predictions DataFrame, per-model
        (y_true, y_pred) pairs for plotting, and the best model name.
    """
    df = load_insurance(config.INSURANCE_PATH)
    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )

    preprocessor = build_classification_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    cv = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)

    rows = []
    predictions = {"y_true": y_test.values}
    plot_data = {}
    cv_scores_by_model = {}
    fitted_pipelines = {}

    for name, model in get_models().items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring=MODEL_SELECTION_METRIC
        )
        cv_scores_by_model[name] = cv_scores.mean()

        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        y_pred = pipeline.predict(X_test)
        metrics = evaluate_regressor(y_test.values, y_pred)
        metrics["model"] = name
        metrics["cv_r2_mean"] = cv_scores_by_model[name]
        rows.append(metrics)

        predictions[f"{name}_pred"] = y_pred
        plot_data[name] = (y_test.values, y_pred)

    metrics_df = pd.DataFrame(rows)[["model", "mae", "rmse", "r2", "cv_r2_mean"]]

    best_model_name = max(cv_scores_by_model, key=cv_scores_by_model.get)
    best_pipeline = fitted_pipelines[best_model_name]

    predictions_df = pd.DataFrame(predictions)

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(config.METRICS_DIR / "regression_metrics.csv", index=False)
    predictions_df.to_csv(config.PREDICTIONS_DIR / "regression_predictions.csv", index=False)
    joblib.dump(best_pipeline, config.MODELS_DIR / "best_regression_model.joblib")

    return {
        "metrics_df": metrics_df,
        "predictions_df": predictions_df,
        "plot_data": plot_data,
        "best_model_name": best_model_name,
        "selection_metric": "mean 5-fold CV R^2 (computed on training data only)",
    }


def run_regression_experiment(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model_names: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run a regression experiment on an arbitrary dataset/feature/target
    selection, as used by the interactive app. Does not write to disk.

    Args:
        df: Source DataFrame.
        feature_columns: Columns to use as model input.
        target_column: Numeric column to predict.
        model_names: Subset of `get_models()` keys to train; all if None.
        test_size: Fraction of data held out for testing.
        random_state: Seed for the split, CV folds, and stochastic estimators.

    Returns:
        Dictionary with metrics_df, predictions_df, plot_data (per-model
        (y_true, y_pred) pairs), fitted_pipelines, best_model_name, and
        selection_metric.

    Raises:
        ValueError: If no feature columns/models are given or the target is
            not numeric.
    """
    if not feature_columns:
        raise ValueError("Select at least one feature column before running an experiment.")
    if model_names is not None and len(model_names) == 0:
        raise ValueError("Select at least one algorithm before running an experiment.")
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise ValueError(
            f"Target column '{target_column}' is not numeric; regression requires a "
            "continuous target."
        )

    validate_no_missing(df, list(feature_columns) + [target_column])

    X = df[feature_columns]
    y = df[target_column]

    numeric_features, categorical_features = infer_feature_types(df, feature_columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = build_classification_preprocessor(numeric_features, categorical_features)

    all_models = get_models()
    if model_names:
        unknown = [name for name in model_names if name not in all_models]
        if unknown:
            raise ValueError(f"Unknown algorithm(s): {unknown}. Available: {list(all_models)}")
        models = {name: all_models[name] for name in model_names}
    else:
        models = all_models

    n_splits = max(2, min(5, len(X_train)))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rows = []
    predictions = {"y_true": y_test.values}
    plot_data = {}
    cv_scores_by_model = {}
    fitted_pipelines = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        cv_score_mean = None
        try:
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="r2")
            cv_score_mean = float(cv_scores.mean())
        except ValueError:
            cv_score_mean = None
        cv_scores_by_model[name] = cv_score_mean

        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        y_pred = pipeline.predict(X_test)
        metrics = evaluate_regressor(y_test.values, y_pred)
        metrics["model"] = name
        metrics["cv_score_mean"] = cv_score_mean
        rows.append(metrics)

        predictions[f"{name}_pred"] = y_pred
        plot_data[name] = (y_test.values, y_pred)

    metrics_df = pd.DataFrame(rows)[["model", "mae", "rmse", "r2", "cv_score_mean"]]

    valid_cv = {k: v for k, v in cv_scores_by_model.items() if v is not None}
    if valid_cv:
        best_model_name = max(valid_cv, key=valid_cv.get)
    else:
        best_model_name = metrics_df.sort_values("r2", ascending=False)["model"].iloc[0]

    predictions_df = pd.DataFrame(predictions)

    return {
        "metrics_df": metrics_df,
        "predictions_df": predictions_df,
        "plot_data": plot_data,
        "fitted_pipelines": fitted_pipelines,
        "best_model_name": best_model_name,
        "selection_metric": f"mean {n_splits}-fold CV R^2 (computed on training data only)",
    }


if __name__ == "__main__":
    result = run_regression()
    print(result["metrics_df"])
    print(f"Best model (by {result['selection_metric']}): {result['best_model_name']}")
