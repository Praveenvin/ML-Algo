"""Run the full project end to end: EDA, classification, clustering, boosting, regression."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.exploratory_analysis import run_eda
from scripts.run_boosting import main as run_boosting_main
from scripts.run_classification import main as run_classification_main
from scripts.run_clustering import main as run_clustering_main
from scripts.run_regression import main as run_regression_main


def main() -> None:
    print("=" * 70)
    print("Step 1/5: Exploratory Data Analysis")
    print("=" * 70)
    run_eda()

    print("\n" + "=" * 70)
    print("Step 2/5: Module A - Classification")
    print("=" * 70)
    run_classification_main()

    print("\n" + "=" * 70)
    print("Step 3/5: Module B - Clustering")
    print("=" * 70)
    run_clustering_main()

    print("\n" + "=" * 70)
    print("Step 4/5: Module C - Boosting")
    print("=" * 70)
    run_boosting_main()

    print("\n" + "=" * 70)
    print("Step 5/5: Module D - Regression (optional)")
    print("=" * 70)
    run_regression_main()

    print("\nAll modules completed. Results saved under results/, models under models/.")


if __name__ == "__main__":
    main()
