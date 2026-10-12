"""Run Module B: Clustering, and save the elbow and cluster scatter plots."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.clustering import run_clustering
from src.visualization import plot_customer_clusters, plot_elbow_and_silhouette


def main() -> None:
    result = run_clustering()

    plot_elbow_and_silhouette(
        result["k_values"],
        result["inertias"],
        result["silhouettes"],
        config.PLOTS_DIR / "clustering_elbow.png",
    )
    plot_customer_clusters(
        result["assignments_df"],
        "Annual Income (k$)",
        "Spending Score (1-100)",
        {
            "K-Means": "KMeans_cluster",
            "Agglomerative": "Agglomerative_cluster",
            "DBSCAN": "DBSCAN_cluster",
        },
        config.PLOTS_DIR / "customer_clusters.png",
    )

    print(result["metrics_df"].to_string(index=False))
    print(f"\nBest K (by silhouette score): {result['best_k']}")
    print(f"DBSCAN eps={result['dbscan_eps']:.3f}, min_samples={result['dbscan_min_samples']}")
    print("\nKMeans segment profiles (mean feature values per cluster):")
    print(result["segment_profiles"].to_string())


if __name__ == "__main__":
    main()
