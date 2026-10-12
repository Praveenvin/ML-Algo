"""Exploratory data analysis across all three datasets."""

from typing import Any

from src import config
from src.data_loader import load_insurance, load_mall_customers, load_social_network_ads
from src.visualization import plot_dataset_overview


def run_eda() -> dict[str, Any]:
    """Load all datasets and generate the dataset overview figure.

    Returns:
        Dictionary with the three loaded DataFrames and the overview Figure,
        so callers (e.g. the EDA notebook) can inspect or display it without
        reloading the data or rebuilding the plot.
    """
    social_df = load_social_network_ads(config.SOCIAL_NETWORK_ADS_PATH)
    mall_df = load_mall_customers(config.MALL_CUSTOMERS_PATH)
    insurance_df = load_insurance(config.INSURANCE_PATH)

    out_path = config.PLOTS_DIR / "dataset_overview.png"
    fig = plot_dataset_overview(social_df, mall_df, insurance_df, out_path)

    print("Social_Network_Ads.csv:")
    print(f"  rows={len(social_df)}, columns={list(social_df.columns)}")
    print(f"  Purchased distribution:\n{social_df['Purchased'].value_counts().to_dict()}")
    print()
    print("Mall_Customers.csv:")
    print(f"  rows={len(mall_df)}, columns={list(mall_df.columns)}")
    print()
    print("insurance_pre.csv:")
    print(f"  rows={len(insurance_df)}, columns={list(insurance_df.columns)}")
    print(f"  Saved overview plot to {out_path}")

    return {
        "social_df": social_df,
        "mall_df": mall_df,
        "insurance_df": insurance_df,
        "overview_fig": fig,
    }


if __name__ == "__main__":
    run_eda()
