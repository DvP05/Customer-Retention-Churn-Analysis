"""
Canonical Metric Layer for Customer Retention & Churn Intelligence.
Provides single-source-of-truth calculations for churn prevalence, retention complement,
tenure metrics, and recurring monthly charge exposure proxy, with DuckDB SQL reconciliation.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import duckdb
import pandas as pd

from retention.config import config


def churn_prevalence(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Computes dataset-level churn prevalence:
    Numerator = labeled churned customers ('Yes')
    Denominator = all customers with valid churn label in snapshot.
    """
    col = target_col or config.target_column
    if df.empty or col not in df.columns:
        return {"churned_count": 0, "labeled_count": 0, "prevalence": 0.0, "prevalence_pct": 0.0}

    labeled = df[col].dropna()
    total = len(labeled)
    if total == 0:
        return {"churned_count": 0, "labeled_count": 0, "prevalence": 0.0, "prevalence_pct": 0.0}

    churned = int((labeled == config.positive_label).sum())
    prevalence = churned / total

    return {
        "churned_count": churned,
        "labeled_count": total,
        "prevalence": float(prevalence),
        "prevalence_pct": round(prevalence * 100, 2),
    }


def retention_complement(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """Computes retention complement (1 - churn prevalence) for the same population."""
    base = churn_prevalence(df, target_col)
    retained_count = base["labeled_count"] - base["churned_count"]
    comp = 1.0 - base["prevalence"] if base["labeled_count"] > 0 else 0.0
    return {
        "retained_count": retained_count,
        "labeled_count": base["labeled_count"],
        "retention_complement": float(comp),
        "retention_complement_pct": round(comp * 100, 2),
    }


def monthly_charge_exposure_proxy(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    charge_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculates the recurring monthly charge exposure proxy:
    Sum of MonthlyCharges for churned customers vs total active base.
    Caveat: This is an exposure proxy, not recognized MRR or GAAP revenue loss.
    """
    t_col = target_col or config.target_column
    c_col = charge_col or config.monthly_charge_column

    if df.empty or t_col not in df.columns or c_col not in df.columns:
        return {
            "churn_exposure_proxy": 0.0,
            "total_monthly_charges": 0.0,
            "exposure_share_pct": 0.0,
        }

    total_charges = float(df[c_col].sum())
    churn_mask = df[t_col] == config.positive_label
    churn_exposure = float(df.loc[churn_mask, c_col].sum())
    share = (churn_exposure / total_charges * 100) if total_charges > 0 else 0.0

    return {
        "churn_exposure_proxy": round(churn_exposure, 2),
        "total_monthly_charges": round(total_charges, 2),
        "exposure_share_pct": round(share, 2),
    }


def calculate_headline_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes all headline executive metrics in a standardized dictionary."""
    prev = churn_prevalence(df)
    ret = retention_complement(df)
    expo = monthly_charge_exposure_proxy(df)

    median_tenure = float(df["tenure"].median()) if "tenure" in df.columns else 0.0
    mean_tenure = float(df["tenure"].mean()) if "tenure" in df.columns else 0.0

    return {
        "total_customers": prev["labeled_count"],
        "churned_customers": prev["churned_count"],
        "churn_prevalence": prev["prevalence"],
        "churn_prevalence_pct": prev["prevalence_pct"],
        "retained_customers": ret["retained_count"],
        "retention_complement": ret["retention_complement"],
        "retention_complement_pct": ret["retention_complement_pct"],
        "churn_exposure_proxy": expo["churn_exposure_proxy"],
        "total_monthly_charges": expo["total_monthly_charges"],
        "exposure_share_pct": expo["exposure_share_pct"],
        "median_tenure_months": round(median_tenure, 1),
        "mean_tenure_months": round(mean_tenure, 1),
    }


def verify_with_duckdb(parquet_path: Path) -> pd.DataFrame:
    """Verifies metrics using an independent DuckDB SQL engine for cross-platform auditability."""
    con = duckdb.connect(database=":memory:")
    # Replace backslashes with forward slashes for DuckDB path parsing
    clean_path = str(parquet_path.resolve()).replace("\\", "/")
    query = f"""
    SELECT
        COUNT(*) AS total_customers,
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
        ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) AS churn_prevalence_pct,
        ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END), 2) AS monthly_exposure_proxy
    FROM read_parquet('{clean_path}')
    """
    res = con.execute(query).df()
    con.close()
    return res


def main() -> None:
    parquet_path = config.processed_parquet
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Clean dataset not found at {parquet_path}. Please run `python -m retention.data.clean` first."
        )

    print(f"Loading processed dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    print("Computing canonical churn and retention metrics...")
    kpis = calculate_headline_kpis(df)

    print("Verifying via DuckDB SQL engine...")
    duckdb_df = verify_with_duckdb(parquet_path)

    print("=" * 60)
    print("CANONICAL HEADLINE KPIS")
    print("=" * 60)
    print(f"Total Customers:            {kpis['total_customers']:,}")
    print(f"Churned Customers:          {kpis['churned_customers']:,}")
    print(f"Dataset Churn Prevalence:   {kpis['churn_prevalence_pct']}%")
    print(f"Retention Complement:       {kpis['retention_complement_pct']}%")
    print(f"Monthly Charge Exposure:    ${kpis['churn_exposure_proxy']:,.2f}")
    print(f"Exposure Share of Total:    {kpis['exposure_share_pct']}%")
    print(f"Median Tenure:              {kpis['median_tenure_months']} months")
    print(f"Mean Tenure:                {kpis['mean_tenure_months']} months")
    print("=" * 60)
    print("DUCKDB SQL AUDIT RESULT:")
    print(duckdb_df.to_string(index=False))
    print("=" * 60)

    # Export KPI table
    table_dir = config.tables_dir
    table_dir.mkdir(parents=True, exist_ok=True)
    kpi_df = pd.DataFrame([kpis])
    kpi_file = table_dir / "churn_kpi_summary.csv"
    kpi_df.to_csv(kpi_file, index=False)
    print(f"Exported KPI summary to {kpi_file}")


if __name__ == "__main__":
    main()
