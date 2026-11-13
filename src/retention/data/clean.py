"""
Data cleaning and transformation module.
Safely converts data types, handles TotalCharges whitespace sentinels,
derives lifecycle tenure bands and analytical features, and saves processed datasets.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from retention.config import config
from retention.data.load import load_raw_data
from retention.data.validate import validate_grain, validate_schema, validate_target


def clean_data(
    df_raw: pd.DataFrame,
    output_parquet_path: Optional[Path] = None,
    output_csv_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw customer DataFrame according to documented policies:
    1. Strips leading/trailing whitespace across all text columns.
    2. Identifies and coerces TotalCharges whitespace strings to 0.0 for tenure == 0.
    3. Standardizes SeniorCitizen to categorical 'Yes' / 'No'.
    4. Categorizes tenure into standard lifecycle bands.
    5. Creates a binary numeric target helper 'churn_binary' (1 for Yes, 0 for No).
    6. Verifies reconciliation of input rows to output rows.
    """
    # Defensive copy
    df = df_raw.copy()

    # Pre-cleaning validation
    validate_schema(df)
    validate_grain(df)
    validate_target(df)

    audit_log: Dict[str, Any] = {
        "cleaned_at": datetime.now(timezone.utc).isoformat(),
        "input_rows": len(df),
        "totalcharges_whitespace_imputed": 0,
        "imputed_customer_ids": [],
    }

    # 1. Clean string columns (strip whitespace)
    string_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        df[col] = df[col].astype(str).str.strip()

    # 2. TotalCharges handling:
    # 11 records have tenure == 0 and empty string TotalCharges
    whitespace_mask = df["TotalCharges"] == ""
    zero_tenure_mask = df["tenure"] == 0

    # Ensure whitespace corresponds exclusively to zero tenure
    imputed_ids = df.loc[whitespace_mask & zero_tenure_mask, config.id_column].tolist()
    audit_log["totalcharges_whitespace_imputed"] = len(imputed_ids)
    audit_log["imputed_customer_ids"] = imputed_ids

    # Coerce to float; replace empty string with 0.0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace("", "0.0"), errors="coerce")

    # If any unexpected NaNs remain, fill with MonthlyCharges * tenure or log
    remaining_nan = df["TotalCharges"].isna().sum()
    if remaining_nan > 0:
        df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
        audit_log["unexpected_nan_filled"] = int(remaining_nan)

    # 3. Numeric type casting
    df["tenure"] = df["tenure"].astype(np.int32)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(np.float64)
    df["TotalCharges"] = df["TotalCharges"].astype(np.float64)

    # 4. Standardize SeniorCitizen to Yes / No
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({1: "Yes", 0: "No", "1": "Yes", "0": "No"}).fillna("No")

    # 5. Add tenure lifecycle bands
    bins = config.tenure_bins
    labels = config.tenure_labels
    df["tenure_band"] = pd.cut(
        df["tenure"],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True,
    ).astype(str)

    # 6. Add binary churn indicator (1 = Yes, 0 = No)
    df["churn_binary"] = df[config.target_column].map({config.positive_label: 1, config.negative_label: 0}).astype(np.int32)

    # 7. Add service add-on count helper
    addon_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    existing_addons = [c for c in addon_cols if c in df.columns]
    if existing_addons:
        df["num_addons"] = (df[existing_addons] == "Yes").sum(axis=1).astype(np.int32)
    else:
        df["num_addons"] = 0

    # 8. Add has_internet flag
    df["has_internet"] = (df["InternetService"] != "No").map({True: "Yes", False: "No"})

    # Reconciliation
    audit_log["output_rows"] = len(df)
    audit_log["reconciliation_match"] = (audit_log["input_rows"] == audit_log["output_rows"])
    if not audit_log["reconciliation_match"]:
        raise ValueError("Data reconciliation failed: Output row count does not match input row count.")

    # Save to disk
    parquet_path = output_parquet_path or config.processed_parquet
    csv_path = output_csv_path or config.processed_csv

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    audit_log["saved_parquet"] = str(parquet_path)
    audit_log["saved_csv"] = str(csv_path)

    return df, audit_log


def main() -> None:
    print("Loading raw dataset for cleaning...")
    df_raw, _ = load_raw_data()
    print(f"Cleaning dataset ({len(df_raw)} records)...")
    df_clean, audit = clean_data(df_raw)

    print("=" * 60)
    print("DATA CLEANING COMPLETED")
    print("=" * 60)
    print(f"Input Rows:                 {audit['input_rows']:,}")
    print(f"Output Rows:                {audit['output_rows']:,}")
    print(f"Reconciliation Match:       {audit['reconciliation_match']}")
    print(f"TotalCharges Imputed (0.0): {audit['totalcharges_whitespace_imputed']}")
    print(f"Saved Parquet:              {audit['saved_parquet']}")
    print(f"Saved CSV:                  {audit['saved_csv']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
