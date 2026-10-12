"""Module A: Classification on Social_Network_Ads.csv."""

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.data_loader import load_social_network_ads
from src.evaluation import compute_confusion_matrix, evaluate_classifier
from src.preprocessing import build_classification_preprocessor, validate_no_missing

NUMERIC_FEATURES = ["Age", "EstimatedSalary"]
CATEGORICAL_FEATURES = ["Gender"]

MODEL_SELECTION_METRIC = "f1"


def get_models() -> dict[str, Any]:
    """Return the classifiers evaluated in this module, keyed by display name."""
    return {
        "Logistic Regression": LogisticRegression(random_state=config.RANDOM_SEED, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=config.RANDOM_SEED),
        "Random Forest": RandomForestClassifier(random_state=config.RANDOM_SEED, n_estimators=200),
        "SVM": CalibratedClassifierCV(
            SVC(random_state=config.RANDOM_SEED), ensemble=False
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate and split the Social_Network_Ads dataframe into X, y."""
    validate_no_missing(df, NUMERIC_FEATURES + CATEGORICAL_FEATURES + [config.CLASSIFICATION_TARGET])
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[config.CLASSIFICATION_TARGET]
    return X, y


def run_classification() -> dict[str, Any]:
    """Train and evaluate all classification models.

    Returns:
        Dictionary with metrics DataFrame, confusion matrices, predictions
        DataFrame, and the name of the best model.
    """
    df = load_social_network_ads(config.SOCIAL_NETWORK_ADS_PATH)
    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
    )

    preprocessor = build_classification_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)

    rows = []
    confusion_matrices = {}
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

        confusion_matrices[name] = compute_confusion_matrix(y_test.values, y_pred)
        predictions[f"{name}_pred"] = y_pred

    metrics_df = pd.DataFrame(rows)[
        ["model", "accuracy", "precision", "recall", "f1_score", "roc_auc", "cv_f1_mean"]
    ]

    best_model_name = max(cv_scores_by_model, key=cv_scores_by_model.get)
    best_pipeline = fitted_pipelines[best_model_name]

    predictions_df = pd.DataFrame(predictions)

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(config.METRICS_DIR / "classification_metrics.csv", index=False)
    predictions_df.to_csv(config.PREDICTIONS_DIR / "classification_predictions.csv", index=False)
    joblib.dump(best_pipeline, config.MODELS_DIR / "best_classification_model.joblib")

    return {
        "metrics_df": metrics_df,
        "confusion_matrices": confusion_matrices,
        "predictions_df": predictions_df,
        "best_model_name": best_model_name,
        "selection_metric": "mean 5-fold CV F1-score (computed on training data only)",
    }


if __name__ == "__main__":
    result = run_classification()
    print(result["metrics_df"])
    print(f"Best model (by {result['selection_metric']}): {result['best_model_name']}")
