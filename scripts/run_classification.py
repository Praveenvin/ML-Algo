"""Run Module A: Classification, and save the model comparison / confusion matrix plots."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.classification import run_classification
from src.visualization import plot_confusion_matrices, plot_model_comparison


def main() -> None:
    result = run_classification()

    plot_model_comparison(
        result["metrics_df"],
        "Classification Model Comparison",
        config.PLOTS_DIR / "classification_comparison.png",
    )
    plot_confusion_matrices(
        result["confusion_matrices"], config.PLOTS_DIR / "confusion_matrices.png"
    )

    print(result["metrics_df"].to_string(index=False))
    print(f"\nBest model (by {result['selection_metric']}): {result['best_model_name']}")


if __name__ == "__main__":
    main()
