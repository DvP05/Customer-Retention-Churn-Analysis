"""
Customer segmentation analysis module.
Calculates churn prevalence, group volume, monthly recurring charge exposure,
and Wilson score 95% confidence intervals across demographic, contract, and service segments.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from retention.config import config


def calculate_wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Computes Wilson score interval for a binomial proportion.
    More reliable than normal approximation, especially for extreme rates or small sample sizes.
    """
    if n == 0:
        return 0.0, 0.0
    p_hat = k / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + (z**2) / n
    centre_adj = p_hat + (z**2) / (2 * n)
    margin = z * np.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * n)) / n)
    lower = max(0.0, (centre_adj - margin) / denom)
    upper = min(1.0, (centre_adj + margin) / denom)
    return float(lower), float(upper)


def analyze_segment(
    df: pd.DataFrame,
    segment_col: str,
    target_col: Optional[str] = None,
    charge_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Computes detailed churn metrics for a given categorical segmentation attribute.
    """
    t_col = target_col or config.target_column
    c_col = charge_col or config.monthly_charge_column

    groups = []
    for val, subset in df.groupby(segment_col, observed=True):
        n = len(subset)
        k = int((subset[t_col] == config.positive_label).sum())
        churn_rate = (k / n) if n > 0 else 0.0
        ci_lower, ci_upper = calculate_wilson_ci(k, n)
        exposure = float(subset.loc[subset[t_col] == config.positive_label, c_col].sum())
        total_charges = float(subset[c_col].sum())

        groups.append({
            "segment_dimension": segment_col,
            "segment_value": str(val),
            "customer_count": n,
            "churned_count": k,
            "churn_rate_pct": round(churn_rate * 100, 2),
            "ci_95_lower_pct": round(ci_lower * 100, 2),
            "ci_95_upper_pct": round(ci_upper * 100, 2),
            "exposure_proxy": round(exposure, 2),
            "total_monthly_charges": round(total_charges, 2),
            "low_sample_flag": bool(n < 30),
        })

    res_df = pd.DataFrame(groups).sort_values("churn_rate_pct", ascending=False)
    return res_df


def run_full_segmentation(
    df: pd.DataFrame,
    dimensions: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Runs segmentation analysis across all key business dimensions."""
    dims = dimensions or [
        "Contract", "InternetService", "PaymentMethod",
        "PaperlessBilling", "TechSupport", "OnlineSecurity",
        "tenure_band", "SeniorCitizen"
    ]

    results = []
    for dim in dims:
        if dim in df.columns:
            seg_df = analyze_segment(df, dim)
            results.append(seg_df)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def main() -> None:
    parquet_path = config.processed_parquet
    if not parquet_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {parquet_path}.")

    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    print("Running segment churn analysis with Wilson 95% confidence intervals...")
    seg_summary = run_full_segmentation(df)

    # Export table
    table_dir = config.tables_dir
    table_dir.mkdir(parents=True, exist_ok=True)
    out_file = table_dir / "segment_analysis.csv"
    seg_summary.to_csv(out_file, index=False)

    print("=" * 60)
    print("TOP 10 VULNERABLE SEGMENTS BY CHURN PREVALENCE")
    print("=" * 60)
    top_10 = seg_summary.sort_values("churn_rate_pct", ascending=False).head(10)
    display_cols = ["segment_dimension", "segment_value", "customer_count", "churned_count", "churn_rate_pct", "ci_95_lower_pct", "ci_95_upper_pct", "exposure_proxy"]
    print(top_10[display_cols].to_string(index=False))
    print("=" * 60)
    print(f"Full segment report exported to {out_file}")


if __name__ == "__main__":
    main()
