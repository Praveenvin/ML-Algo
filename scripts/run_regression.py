"""Run optional Module D: Regression, and save the actual-vs-predicted plot."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.regression import run_regression
from src.visualization import plot_actual_vs_predicted


def main() -> None:
    result = run_regression()

    plot_actual_vs_predicted(
        result["plot_data"], config.PLOTS_DIR / "regression_actual_vs_predicted.png"
    )

    print(result["metrics_df"].to_string(index=False))
    print(f"\nBest model (by {result['selection_metric']}): {result['best_model_name']}")


if __name__ == "__main__":
    main()
