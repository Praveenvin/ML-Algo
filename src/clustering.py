"""Module B: Clustering on Mall_Customers.csv."""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from src import config
from src.data_loader import load_mall_customers
from src.evaluation import evaluate_clustering
from src.preprocessing import build_clustering_preprocessor, validate_no_missing

# Benign, Windows-specific sklearn/MKL threading notice (not a correctness
# issue): fires on every KMeans fit, so it would otherwise repeat once per K
# in the elbow-method loop below. Silenced by message so other UserWarnings
# (e.g. genuine convergence warnings) still surface normally.
warnings.filterwarnings(
    "ignore",
    message="KMeans is known to have a memory leak on Windows with MKL.*",
    category=UserWarning,
)

K_RANGE = range(2, 11)
DEFAULT_DBSCAN_EPS = 0.5
DEFAULT_DBSCAN_MIN_SAMPLES = 5


def prepare_data(df: pd.DataFrame) -> np.ndarray:
    """Validate and scale the clustering features."""
    validate_no_missing(df, config.CLUSTERING_FEATURES)
    X = df[config.CLUSTERING_FEATURES].values
    scaler = build_clustering_preprocessor(config.CLUSTERING_FEATURES)
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def run_elbow_and_silhouette_analysis(X_scaled: np.ndarray) -> tuple[list[int], list[float], list[float | None]]:
    """Run K-Means across K_RANGE, collecting inertia and silhouette score per K."""
    inertias = []
    silhouettes = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=config.RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        metrics = evaluate_clustering(X_scaled, labels)
        silhouettes.append(metrics["silhouette_score"])
    return list(K_RANGE), inertias, silhouettes


def choose_best_k(k_values: list[int], silhouettes: list[float | None]) -> int:
    """Pick the K with the highest silhouette score among valid candidates."""
    valid = [(k, s) for k, s in zip(k_values, silhouettes) if s is not None]
    if not valid:
        raise ValueError("No valid silhouette scores were computed across the K range.")
    return max(valid, key=lambda pair: pair[1])[0]


def estimate_dbscan_eps(X_scaled: np.ndarray, min_samples: int) -> float:
    """Estimate a reasonable DBSCAN eps using the k-distance elbow heuristic."""
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])
    return float(np.percentile(k_distances, 90))


def run_clustering() -> dict[str, Any]:
    """Run K-Means, Agglomerative, and DBSCAN clustering and save results.

    Returns:
        Dictionary with cluster assignments DataFrame, metrics DataFrame,
        elbow/silhouette curves, and per-segment characterizations.
    """
    df = load_mall_customers(config.MALL_CUSTOMERS_PATH)
    X_scaled = prepare_data(df)

    k_values, inertias, silhouettes = run_elbow_and_silhouette_analysis(X_scaled)
    best_k = choose_best_k(k_values, silhouettes)

    kmeans = KMeans(n_clusters=best_k, random_state=config.RANDOM_SEED, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    agglomerative = AgglomerativeClustering(n_clusters=best_k)
    agglomerative_labels = agglomerative.fit_predict(X_scaled)

    dbscan_min_samples = DEFAULT_DBSCAN_MIN_SAMPLES
    dbscan_eps = estimate_dbscan_eps(X_scaled, dbscan_min_samples)
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    results_rows = []
    for name, labels in [
        ("KMeans", kmeans_labels),
        ("Agglomerative", agglomerative_labels),
        ("DBSCAN", dbscan_labels),
    ]:
        metrics = evaluate_clustering(X_scaled, labels)
        metrics["model"] = name
        if name == "KMeans":
            metrics["chosen_k"] = best_k
        elif name == "Agglomerative":
            metrics["chosen_k"] = best_k
        else:
            metrics["chosen_k"] = None
        results_rows.append(metrics)

    metrics_df = pd.DataFrame(results_rows)[
        ["model", "n_clusters", "n_noise_points", "silhouette_score", "chosen_k"]
    ]

    assignments_df = df.copy()
    assignments_df["KMeans_cluster"] = kmeans_labels
    assignments_df["Agglomerative_cluster"] = agglomerative_labels
    assignments_df["DBSCAN_cluster"] = dbscan_labels

    segment_profiles = (
        assignments_df.groupby("KMeans_cluster")[config.CLUSTERING_FEATURES + ["Age"]]
        .mean()
        .round(1)
    )
    segment_profiles["count"] = assignments_df.groupby("KMeans_cluster").size()

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(config.METRICS_DIR / "clustering_metrics.csv", index=False)
    assignments_df.to_csv(config.CLUSTER_ASSIGNMENTS_DIR / "customer_clusters.csv", index=False)

    return {
        "metrics_df": metrics_df,
        "assignments_df": assignments_df,
        "k_values": k_values,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "best_k": best_k,
        "dbscan_eps": dbscan_eps,
        "dbscan_min_samples": dbscan_min_samples,
        "segment_profiles": segment_profiles,
    }


ALGORITHM_FACTORY_NAMES = ("KMeans", "Agglomerative", "DBSCAN")


def run_clustering_experiment(
    df: pd.DataFrame,
    feature_columns: list[str],
    algorithms: list[str] | None = None,
    kmeans_k: int = 5,
    agglomerative_k: int = 5,
    dbscan_eps: float | None = None,
    dbscan_min_samples: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run clustering on an arbitrary dataset/feature selection, as used by
    the interactive app. Does not write to disk.

    Args:
        df: Source DataFrame.
        feature_columns: Numeric columns to cluster on (categorical columns
            are rejected, since Euclidean-distance clustering requires a
            numeric feature space).
        algorithms: Subset of "KMeans", "Agglomerative", "DBSCAN" to run;
            all three if None.
        kmeans_k: Number of clusters for K-Means.
        agglomerative_k: Number of clusters for Agglomerative Clustering.
        dbscan_eps: DBSCAN neighborhood radius; estimated via the k-distance
            heuristic if None.
        dbscan_min_samples: DBSCAN minimum samples per neighborhood.
        random_state: Seed for K-Means and PCA.

    Returns:
        Dictionary with metrics_df, assignments_df (original df + one
        `<Algorithm>_cluster` column per algorithm run), scaled feature
        matrix, resolved dbscan_eps, and PCA 2D components (pca_df) when
        more than 2 feature columns were selected, for visualization.

    Raises:
        ValueError: If fewer than 2 numeric feature columns are selected,
            or an unknown algorithm name is given.
    """
    if algorithms is None:
        algorithms = list(ALGORITHM_FACTORY_NAMES)
    unknown = [a for a in algorithms if a not in ALGORITHM_FACTORY_NAMES]
    if unknown:
        raise ValueError(f"Unknown algorithm(s): {unknown}. Available: {list(ALGORITHM_FACTORY_NAMES)}")
    if not algorithms:
        raise ValueError("Select at least one clustering algorithm before running an experiment.")

    numeric_features = [c for c in feature_columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_features) < 2:
        raise ValueError(
            "Clustering requires at least 2 numeric feature columns; "
            f"got {len(numeric_features)} numeric column(s) among the selection."
        )

    validate_no_missing(df, numeric_features)
    X = df[numeric_features].values
    scaler = build_clustering_preprocessor(numeric_features)
    X_scaled = scaler.fit_transform(X)

    assignments_df = df.copy()
    results_rows = []
    resolved_dbscan_eps = dbscan_eps

    if "KMeans" in algorithms:
        kmeans = KMeans(n_clusters=kmeans_k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        assignments_df["KMeans_cluster"] = labels
        metrics = evaluate_clustering(X_scaled, labels)
        metrics["model"] = "KMeans"
        metrics["chosen_k"] = kmeans_k
        results_rows.append(metrics)

    if "Agglomerative" in algorithms:
        agglomerative = AgglomerativeClustering(n_clusters=agglomerative_k)
        labels = agglomerative.fit_predict(X_scaled)
        assignments_df["Agglomerative_cluster"] = labels
        metrics = evaluate_clustering(X_scaled, labels)
        metrics["model"] = "Agglomerative"
        metrics["chosen_k"] = agglomerative_k
        results_rows.append(metrics)

    if "DBSCAN" in algorithms:
        if resolved_dbscan_eps is None:
            resolved_dbscan_eps = estimate_dbscan_eps(X_scaled, dbscan_min_samples)
        dbscan = DBSCAN(eps=resolved_dbscan_eps, min_samples=dbscan_min_samples)
        labels = dbscan.fit_predict(X_scaled)
        assignments_df["DBSCAN_cluster"] = labels
        metrics = evaluate_clustering(X_scaled, labels)
        metrics["model"] = "DBSCAN"
        metrics["chosen_k"] = None
        results_rows.append(metrics)

    metrics_df = pd.DataFrame(results_rows)[
        ["model", "n_clusters", "n_noise_points", "silhouette_score", "chosen_k"]
    ]

    pca_df = None
    pca_explained_variance_ratio = None
    if len(numeric_features) > 2:
        pca = PCA(n_components=2, random_state=random_state)
        components = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame({"PC1": components[:, 0], "PC2": components[:, 1]})
        pca_explained_variance_ratio = tuple(float(v) for v in pca.explained_variance_ratio_)

    return {
        "metrics_df": metrics_df,
        "assignments_df": assignments_df,
        "X_scaled": X_scaled,
        "numeric_features": numeric_features,
        "dbscan_eps": resolved_dbscan_eps,
        "dbscan_min_samples": dbscan_min_samples,
        "pca_df": pca_df,
        "pca_explained_variance_ratio": pca_explained_variance_ratio,
    }


if __name__ == "__main__":
    result = run_clustering()
    print(result["metrics_df"])
    print(f"Best K (by silhouette score): {result['best_k']}")
    print(f"DBSCAN eps={result['dbscan_eps']:.3f}, min_samples={result['dbscan_min_samples']}")
    print("\nKMeans segment profiles:")
    print(result["segment_profiles"])
