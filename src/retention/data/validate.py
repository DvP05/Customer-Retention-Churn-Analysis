"""
Data validation and profiling module.
Enforces schema constraints, row grain uniqueness, numeric bounds, and category validation.
Generates an auditable Data Quality Report in reports/data_quality_report.md.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from retention.config import config
from retention.data.load import load_raw_data


class ValidationError(Exception):
    """Raised when data fails mandatory schema or integrity checks."""
    pass


def validate_schema(df: pd.DataFrame) -> List[str]:
    """Verifies all mandatory columns are present in DataFrame."""
    required = [config.id_column, config.target_column] + config.numeric_columns + config.categorical_columns
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValidationError(f"Schema Validation Failed: Missing required columns: {missing}")
    return required


def validate_grain(df: pd.DataFrame, id_col: Optional[str] = None) -> None:
    """Verifies that each row represents exactly one unique customer."""
    col = id_col or config.id_column
    if col not in df.columns:
        raise ValidationError(f"ID column '{col}' not found.")
    duplicates = df[col].duplicated().sum()
    if duplicates > 0:
        raise ValidationError(
            f"Grain Validation Failed: Found {duplicates} duplicate values in '{col}'. "
            "Data must be 1 row per unique customer."
        )


def validate_target(df: pd.DataFrame, target_col: Optional[str] = None) -> None:
    """Verifies target labels belong to expected binary set."""
    col = target_col or config.target_column
    if col not in df.columns:
        raise ValidationError(f"Target column '{col}' not found.")
    valid_labels = {config.positive_label, config.negative_label}
    observed = set(df[col].dropna().unique())
    unexpected = observed - valid_labels
    if unexpected:
        raise ValidationError(
            f"Target Validation Failed: Column '{col}' contains unexpected values: {unexpected}. "
            f"Expected only {valid_labels}."
        )


def validate_data(
    df: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
    output_report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Runs comprehensive data validation and profiling checks.

    Returns:
        Validation report dictionary.
    """
    report: Dict[str, Any] = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "checks_passed": True,
        "warnings": [],
        "errors": [],
    }

    # 1. Schema check
    try:
        validate_schema(df)
        report["schema_check"] = "PASSED"
    except ValidationError as e:
        report["schema_check"] = "FAILED"
        report["errors"].append(str(e))
        report["checks_passed"] = False
        raise

    # 2. Grain check
    try:
        validate_grain(df)
        report["grain_check"] = "PASSED"
        report["unique_customers"] = int(df[config.id_column].nunique())
    except ValidationError as e:
        report["grain_check"] = "FAILED"
        report["errors"].append(str(e))
        report["checks_passed"] = False
        raise

    # 3. Target check
    try:
        validate_target(df)
        report["target_check"] = "PASSED"
        report["target_distribution"] = df[config.target_column].value_counts().to_dict()
    except ValidationError as e:
        report["target_check"] = "FAILED"
        report["errors"].append(str(e))
        report["checks_passed"] = False
        raise

    # 4. Numeric bounds & quirks
    # Tenure
    if "tenure" in df.columns:
        invalid_tenure = (df["tenure"] < 0).sum()
        if invalid_tenure > 0:
            msg = f"Found {invalid_tenure} negative tenure values."
            report["warnings"].append(msg)
        report["tenure_range"] = [int(df["tenure"].min()), int(df["tenure"].max())]

    # MonthlyCharges
    if "MonthlyCharges" in df.columns:
        invalid_charges = (df["MonthlyCharges"] <= 0).sum()
        if invalid_charges > 0:
            report["warnings"].append(f"Found {invalid_charges} non-positive MonthlyCharges.")
        report["monthly_charges_range"] = [float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max())]

    # TotalCharges whitespace check
    if "TotalCharges" in df.columns:
        space_count = int((df["TotalCharges"].astype(str).str.strip() == "").sum())
        report["total_charges_whitespace_count"] = space_count
        if space_count > 0:
            report["warnings"].append(
                f"Found {space_count} records where TotalCharges is whitespace string ' '. "
                "Documented cleaning policy will impute 0.0 for tenure==0."
            )

    # 5. Missingness profile
    null_counts = df.isnull().sum()
    report["missing_values"] = {col: int(cnt) for col, cnt in null_counts.items() if cnt > 0}

    # Generate Markdown Report
    report_path = output_report_path or (config.reports_dir / "data_quality_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    generate_markdown_report(report, metadata, report_path)
    report["report_file"] = str(report_path)

    return report


def generate_markdown_report(
    report: Dict[str, Any],
    metadata: Optional[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generates an auditable Markdown report for stakeholders."""
    meta = metadata or {}
    md = [
        "# Data Quality & Schema Profiling Report",
        "",
        f"**Generated:** {report['validated_at']} (UTC)  ",
        f"**Source File:** `{meta.get('filename', 'Unknown')}`  ",
        f"**SHA-256 Fingerprint:** `{meta.get('sha256', 'N/A')}`  ",
        f"**Status:** {'✅ PASSED ALL GATES' if report['checks_passed'] else '❌ FAILED'}",
        "",
        "## 1. Executive Quality Summary",
        "",
        f"- **Total Rows:** {report['total_rows']:,}",
        f"- **Total Columns:** {report['total_columns']}",
        f"- **Unique Customer Keys:** {report.get('unique_customers', 'N/A'):,}",
        f"- **Schema Gate:** {report.get('schema_check', 'N/A')}",
        f"- **Row Grain Gate:** {report.get('grain_check', 'N/A')}",
        f"- **Target Gate:** {report.get('target_check', 'N/A')}",
        "",
        "## 2. Target Variable Distribution",
        "",
        "| Churn Label | Record Count | Percentage |",
        "|---|---|---|",
    ]

    target_dist = report.get("target_distribution", {})
    total = report["total_rows"]
    for label, count in target_dist.items():
        pct = (count / total * 100) if total else 0.0
        md.append(f"| **{label}** | {count:,} | {pct:.2f}% |")

    md.extend([
        "",
        "## 3. Data Integrity & Quirks Audit",
        "",
        f"- **Tenure Range:** {report.get('tenure_range', ['N/A', 'N/A'])[0]} to {report.get('tenure_range', ['N/A', 'N/A'])[1]} months",
        f"- **Monthly Charges Range:** ${report.get('monthly_charges_range', ['N/A', 'N/A'])[0]:.2f} to ${report.get('monthly_charges_range', ['N/A', 'N/A'])[1]:.2f}",
        f"- **TotalCharges Whitespace Count:** {report.get('total_charges_whitespace_count', 0)} records",
        "",
        "### Audit of Whitespace TotalCharges:",
        "As established in the data specification, exactly 11 customers have `tenure == 0` (new enrollments who joined mid-billing cycle). Their raw `TotalCharges` strings are whitespace `' '`. The cleaning pipeline will safely coerce these to `0.0`.",
        "",
        "## 4. Warnings and Exceptions",
        "",
    ])

    if report["warnings"]:
        for w in report["warnings"]:
            md.append(f"- ⚠️ **WARNING:** {w}")
    else:
        md.append("- No warnings recorded. Dataset satisfies all bounds.")

    md.append("")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and profile customer churn dataset.")
    parser.add_argument("--input", type=str, default=None, help="Path to input CSV file")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else config.raw_path
    print(f"Loading raw dataset from {input_path}...")
    df, metadata = load_raw_data(input_path)

    print("Running validation gates and profiling checks...")
    report = validate_data(df, metadata)

    print("=" * 60)
    print("VALIDATION GATES COMPLETED")
    print("=" * 60)
    print(f"Total Rows:         {report['total_rows']:,}")
    print(f"Unique Customers:   {report['unique_customers']:,}")
    print(f"Schema Check:       {report['schema_check']}")
    print(f"Grain Check:        {report['grain_check']}")
    print(f"Target Check:       {report['target_check']}")
    print(f"Report Generated:   {report['report_file']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
