"""
Unit tests for data schema validation, grain checks, and error handling.
"""

import pandas as pd
import pytest

from retention.data.validate import (
    ValidationError,
    validate_data,
    validate_grain,
    validate_schema,
    validate_target,
)


def test_validate_schema_success(sample_raw_data: pd.DataFrame) -> None:
    """Verifies that sample raw data with all required columns passes schema check."""
    required = validate_schema(sample_raw_data)
    assert len(required) >= 21


def test_validate_schema_missing_column(sample_raw_data: pd.DataFrame) -> None:
    """Verifies that missing mandatory column raises ValidationError."""
    df_missing = sample_raw_data.drop(columns=["MonthlyCharges"])
    with pytest.raises(ValidationError, match="Missing required columns"):
        validate_schema(df_missing)


def test_validate_grain_duplicate_id(sample_raw_data: pd.DataFrame) -> None:
    """Verifies that duplicate customer keys trigger a grain validation failure."""
    df_dup = sample_raw_data.copy()
    df_dup.loc[1, "customerID"] = df_dup.loc[0, "customerID"]
    with pytest.raises(ValidationError, match="duplicate values in 'customerID'"):
        validate_grain(df_dup)


def test_validate_target_unexpected_label(sample_raw_data: pd.DataFrame) -> None:
    """Verifies that unexpected target values raise ValidationError."""
    df_bad_target = sample_raw_data.copy()
    df_bad_target.loc[0, "Churn"] = "Maybe"
    with pytest.raises(ValidationError, match="contains unexpected values"):
        validate_target(df_bad_target)


def test_validate_data_generates_report(sample_raw_data: pd.DataFrame, tmp_path) -> None:
    """Verifies that validate_data runs all gates and outputs report."""
    report_file = tmp_path / "test_report.md"
    report = validate_data(sample_raw_data, output_report_path=report_file)
    assert report["checks_passed"] is True
    assert report["total_rows"] == 10
    assert report_file.exists()
