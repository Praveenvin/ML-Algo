"""Run Module C: Boosting, compare with Module A classifiers, and save plots."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.boosting import compare_with_classification, run_boosting
from src.classification import run_classification
from src.visualization import plot_model_comparison


def main() -> None:
    classification_result = run_classification()
    boosting_result = run_boosting()

    plot_model_comparison(
        boosting_result["metrics_df"],
        "Boosting Model Comparison",
        config.PLOTS_DIR / "boosting_comparison.png",
    )

    print(boosting_result["metrics_df"].to_string(index=False))
    print(f"\nBest boosting model (by {boosting_result['selection_metric']}): "
          f"{boosting_result['best_model_name']}")

    combined = compare_with_classification(
        classification_result["metrics_df"], boosting_result["metrics_df"]
    )
    print("\nCombined comparison (standard classifiers vs boosting), sorted by F1-score:")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
