"""
Cohort and lifecycle capability module.
Enforces honest semantics: verifies whether true calendar cohorts are supported by the data,
and computes defensible lifecycle tenure-band aggregations.
"""

from typing import Any, Dict, List, Tuple
import pandas as pd

from retention.config import config


def check_cohort_capability(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Checks if dataset has valid dated event timestamps to support true calendar cohort retention.
    For standard IBM Telco snapshot, returns False with documented justification.
    """
    candidate_date_cols = ["signup_date", "subscription_start", "event_date", "cancellation_date"]
    found_dates = [c for c in candidate_date_cols if c in df.columns]

    if not found_dates:
        explanation = (
            "Static Snapshot Notice: The dataset provides customer tenure in months, but does not "
            "contain individual dated subscription events (e.g., signup dates or monthly activity logs). "
            "Therefore, true calendar signup cohorts cannot be produced without fabricating artificial timestamps. "
            "Analysis is properly focused on empirical lifecycle tenure bands."
        )
        return False, explanation

    return True, f"Found dated event columns: {found_dates}. Cohort analysis supported."


def analyze_tenure_bands(
    df: pd.DataFrame,
    target_col: str = "Churn",
    charge_col: str = "MonthlyCharges",
) -> pd.DataFrame:
    """
    Aggregates customer churn, retention, and charge exposure across defined lifecycle tenure bands.
    """
    if "tenure_band" not in df.columns:
        raise ValueError("Column 'tenure_band' not found in DataFrame.")

    band_rows: List[Dict[str, Any]] = []
    total_customers = len(df)

    for band, subset in df.groupby("tenure_band", observed=True):
        n = len(subset)
        churned = int((subset[target_col] == config.positive_label).sum())
        churn_rate = (churned / n) if n > 0 else 0.0
        retained = n - churned
        retention_rate = (retained / n) if n > 0 else 0.0
        exposure = float(subset.loc[subset[target_col] == config.positive_label, charge_col].sum())
        share_of_total = (n / total_customers * 100) if total_customers > 0 else 0.0

        band_rows.append({
            "tenure_band": str(band),
            "customer_count": n,
            "share_of_total_customers_pct": round(share_of_total, 2),
            "churned_count": churned,
            "retained_count": retained,
            "churn_prevalence_pct": round(churn_rate * 100, 2),
            "retention_rate_pct": round(retention_rate * 100, 2),
            "median_tenure_months": float(subset["tenure"].median()),
            "mean_monthly_charges": round(float(subset[charge_col].mean()), 2),
            "exposure_proxy": round(exposure, 2),
        })

    band_df = pd.DataFrame(band_rows)
    # Sort according to configured tenure labels order if possible
    order_map = {lbl: i for i, lbl in enumerate(config.tenure_labels)}
    band_df["sort_key"] = band_df["tenure_band"].map(order_map).fillna(99)
    band_df = band_df.sort_values("sort_key").drop(columns=["sort_key"])

    return band_df
