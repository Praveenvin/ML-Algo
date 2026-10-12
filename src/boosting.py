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
from src.evaluation import evaluate_classifier
from src.preprocessing import build_classification_preprocessor

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


if __name__ == "__main__":
    result = run_boosting()
    print(result["metrics_df"])
    print(f"Best boosting model (by {result['selection_metric']}): {result['best_model_name']}")
