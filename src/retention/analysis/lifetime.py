"""
Customer lifetime and tenure distribution module.
Calculates tenure distributions, lifecycle summary statistics,
and exports lifecycle tenure band reports.
"""

from typing import Any, Dict
import pandas as pd

from retention.analysis.cohorts import analyze_tenure_bands, check_cohort_capability
from retention.config import config


def compute_tenure_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes summary statistics for customer tenure overall and split by churn status."""
    if "tenure" not in df.columns:
        raise ValueError("Column 'tenure' missing from DataFrame.")

    churned = df[df[config.target_column] == config.positive_label]["tenure"]
    retained = df[df[config.target_column] == config.negative_label]["tenure"]

    return {
        "overall": {
            "mean": round(float(df["tenure"].mean()), 2),
            "std": round(float(df["tenure"].std()), 2),
            "median": float(df["tenure"].median()),
            "q25": float(df["tenure"].quantile(0.25)),
            "q75": float(df["tenure"].quantile(0.75)),
            "iqr": float(df["tenure"].quantile(0.75) - df["tenure"].quantile(0.25)),
            "min": int(df["tenure"].min()),
            "max": int(df["tenure"].max()),
        },
        "churned": {
            "mean": round(float(churned.mean()), 2),
            "median": float(churned.median()),
            "q25": float(churned.quantile(0.25)),
            "q75": float(churned.quantile(0.75)),
        },
        "retained": {
            "mean": round(float(retained.mean()), 2),
            "median": float(retained.median()),
            "q25": float(retained.quantile(0.25)),
            "q75": float(retained.quantile(0.75)),
        },
    }


def compute_empirical_survival_curve(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes empirical survival/retention curve across tenure months.
    Caveat: Snapshot tenure reflects cross-sectional age, not a controlled longitudinal cohort.
    """
    total = len(df)
    max_tenure = int(df["tenure"].max())
    curve = []

    for m in range(0, max_tenure + 1):
        surviving = int((df["tenure"] >= m).sum())
        rate = (surviving / total) if total > 0 else 0.0
        curve.append({
            "tenure_months": m,
            "customers_active_or_longer": surviving,
            "empirical_survival_rate": round(rate, 4),
            "empirical_survival_rate_pct": round(rate * 100, 2),
        })

    return pd.DataFrame(curve)


def main() -> None:
    parquet_path = config.processed_parquet
    if not parquet_path.exists():
        raise FileNotFoundError(f"Clean dataset not found at {parquet_path}.")

    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # Check cohort capability
    is_cohort_ready, notice = check_cohort_capability(df)
    print("\n" + "=" * 60)
    print("COHORT CAPABILITY AUDIT")
    print("=" * 60)
    print(f"Supported: {is_cohort_ready}")
    print(f"Notice:    {notice}")

    # Compute tenure stats
    stats = compute_tenure_statistics(df)
    print("\n" + "=" * 60)
    print("TENURE SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Overall Tenure:  Median = {stats['overall']['median']} mo, Mean = {stats['overall']['mean']} mo, IQR = {stats['overall']['iqr']} mo")
    print(f"Churned Tenure:  Median = {stats['churned']['median']} mo, Mean = {stats['churned']['mean']} mo")
    print(f"Retained Tenure: Median = {stats['retained']['median']} mo, Mean = {stats['retained']['mean']} mo")

    # Analyze tenure bands
    band_df = analyze_tenure_bands(df)
    print("\n" + "=" * 60)
    print("LIFECYCLE TENURE BAND PERFORMANCE")
    print("=" * 60)
    cols = ["tenure_band", "customer_count", "churned_count", "churn_prevalence_pct", "exposure_proxy"]
    print(band_df[cols].to_string(index=False))

    # Export table
    table_dir = config.tables_dir
    table_dir.mkdir(parents=True, exist_ok=True)
    out_file = table_dir / "tenure_band_summary.csv"
    band_df.to_csv(out_file, index=False)
    print(f"\nExported tenure band summary to {out_file}")


if __name__ == "__main__":
    main()
