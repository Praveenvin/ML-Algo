"""Module C: Boosting algorithms on Social_Network_Ads.csv, compared against
the standard classifiers from Module A using the same split and metrics."""

from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src import config
from src.classification import CATEGORICAL_FEATURES, NUMERIC_FEATURES, prepare_data
from src.data_loader import load_social_network_ads
from src.evaluation import compute_confusion_matrix, evaluate_classifier
from src.preprocessing import build_classification_preprocessor, infer_feature_types, validate_no_missing

MODEL_SELECTION_METRIC = "f1"


def get_models() -> dict[str, Any]:
    """Return the boosting classifiers evaluated in this module."""
    return {
        "AdaBoost": AdaBoostClassifier(random_state=config.RANDOM_SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=config.RANDOM_SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=config.RANDOM_SEED),
    }


def run_boosting() -> dict[str, Any]:
    """Train and evaluate all boosting models using the same split/target as Module A.

    Returns:
        Dictionary with metrics DataFrame, predictions DataFrame, and best model name.
    """
    df = load_social_network_ads(config.SOCIAL_NETWORK_ADS_PATH)
    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
    )

    preprocessor = build_classification_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)

    rows = []
    predictions = {"y_true": y_test.values}
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
        y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

        metrics = evaluate_classifier(y_test.values, y_pred, y_proba)
        metrics["model"] = name
        metrics["cv_f1_mean"] = cv_scores_by_model[name]
        rows.append(metrics)

        predictions[f"{name}_pred"] = y_pred

    metrics_df = pd.DataFrame(rows)[
        ["model", "accuracy", "precision", "recall", "f1_score", "roc_auc", "cv_f1_mean"]
    ]

    best_model_name = max(cv_scores_by_model, key=cv_scores_by_model.get)
    best_pipeline = fitted_pipelines[best_model_name]

    predictions_df = pd.DataFrame(predictions)

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(config.METRICS_DIR / "boosting_metrics.csv", index=False)
    predictions_df.to_csv(config.PREDICTIONS_DIR / "boosting_predictions.csv", index=False)
    joblib.dump(best_pipeline, config.MODELS_DIR / "best_boosting_model.joblib")

    return {
        "metrics_df": metrics_df,
        "predictions_df": predictions_df,
        "best_model_name": best_model_name,
        "selection_metric": "mean 5-fold CV F1-score (computed on training data only)",
    }


def compare_with_classification(
    classification_metrics_df: pd.DataFrame, boosting_metrics_df: pd.DataFrame
) -> pd.DataFrame:
    """Combine classification and boosting metrics into a single comparison table."""
    combined = pd.concat([classification_metrics_df, boosting_metrics_df], ignore_index=True)
    return combined.sort_values("f1_score", ascending=False).reset_index(drop=True)


def get_feature_importances(pipeline: Pipeline) -> dict[str, float] | None:
    """Extract feature importances from a fitted boosting pipeline, if supported.

    Args:
        pipeline: A fitted Pipeline with 'preprocessor' and 'model' steps.

    Returns:
        Mapping of post-preprocessing feature name to importance, sorted
        descending, or None if the estimator has no `feature_importances_`.
    """
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None
    try:
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
    importances = dict(zip(feature_names, model.feature_importances_.tolist()))
    return dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))


def run_boosting_experiment(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model_names: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run a boosting experiment on an arbitrary dataset/feature/target
    selection, as used by the interactive app. Does not write to disk.

    Mirrors `src.classification.run_classification_experiment` so that
    results from both can be compared directly (same metric set, same
    distinction between CV-based selection and test-set reporting).

    Returns:
        Dictionary with metrics_df, confusion_matrices, predictions_df,
        fitted_pipelines, feature_importances, best_model_name, y_test,
        n_classes, classes, and selection_metric.
    """
    if not feature_columns:
        raise ValueError("Select at least one feature column before running an experiment.")
    if model_names is not None and len(model_names) == 0:
        raise ValueError("Select at least one algorithm before running an experiment.")

    validate_no_missing(df, list(feature_columns) + [target_column])

    X = df[feature_columns]
    y = df[target_column]
    n_classes = int(y.nunique())
    if n_classes < 2:
        raise ValueError(
            f"Target column '{target_column}' has only {n_classes} unique value(s); "
            "classification requires at least 2 classes."
        )

    classes = sorted(y.unique().tolist())
    numeric_features, categorical_features = infer_feature_types(df, feature_columns)

    class_counts = y.value_counts()
    can_stratify = class_counts.min() >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if can_stratify else None
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

    scoring = "f1" if n_classes == 2 else "f1_weighted"
    min_class_count = int(y_train.value_counts().min()) if can_stratify else len(y_train)
    n_splits = max(2, min(5, min_class_count))
    cv = (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        if can_stratify and min_class_count >= 2
        else None
    )

    rows = []
    confusion_matrices = {}
    probabilities = {}
    feature_importances = {}
    predictions = {"y_true": y_test.values}
    cv_scores_by_model = {}
    fitted_pipelines = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        cv_score_mean = None
        if cv is not None:
            try:
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring)
                cv_score_mean = float(cv_scores.mean())
            except ValueError:
                cv_score_mean = None
        cv_scores_by_model[name] = cv_score_mean

        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline
        feature_importances[name] = get_feature_importances(pipeline)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
        probabilities[name] = y_proba

        metrics = evaluate_classifier(y_test.values, y_pred, y_proba)
        metrics["model"] = name
        metrics["cv_score_mean"] = cv_score_mean
        rows.append(metrics)

        confusion_matrices[name] = compute_confusion_matrix(y_test.values, y_pred, labels=classes)
        predictions[f"{name}_pred"] = y_pred

    metrics_df = pd.DataFrame(rows)[
        ["model", "accuracy", "precision", "recall", "f1_score", "roc_auc", "cv_score_mean"]
    ]

    valid_cv = {k: v for k, v in cv_scores_by_model.items() if v is not None}
    if valid_cv:
        best_model_name = max(valid_cv, key=valid_cv.get)
    else:
        best_model_name = metrics_df.sort_values("f1_score", ascending=False)["model"].iloc[0]

    predictions_df = pd.DataFrame(predictions)

    return {
        "metrics_df": metrics_df,
        "confusion_matrices": confusion_matrices,
        "probabilities": probabilities,
        "predictions_df": predictions_df,
        "fitted_pipelines": fitted_pipelines,
        "feature_importances": feature_importances,
        "best_model_name": best_model_name,
        "y_test": y_test.values,
        "n_classes": n_classes,
        "classes": classes,
        "selection_metric": (
            f"mean {n_splits}-fold CV {scoring} score (computed on training data only)"
            if cv is not None
            else "test-set F1-score (CV unavailable: too few samples per class)"
        ),
    }


if __name__ == "__main__":
    result = run_boosting()
    print(result["metrics_df"])
    print(f"Best boosting model (by {result['selection_metric']}): {result['best_model_name']}")
