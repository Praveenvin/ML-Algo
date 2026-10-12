"""Module B: Clustering on Mall_Customers.csv."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.neighbors import NearestNeighbors

from src import config
from src.data_loader import load_mall_customers
from src.evaluation import evaluate_clustering
from src.preprocessing import build_clustering_preprocessor, validate_no_missing

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


if __name__ == "__main__":
    result = run_clustering()
    print(result["metrics_df"])
    print(f"Best K (by silhouette score): {result['best_k']}")
    print(f"DBSCAN eps={result['dbscan_eps']:.3f}, min_samples={result['dbscan_min_samples']}")
    print("\nKMeans segment profiles:")
    print(result["segment_profiles"])
