"""
Unit tests for data cleaning, TotalCharges handling, and type normalization.
"""

import pandas as pd

from retention.data.clean import clean_data


def test_clean_data_totalcharges_whitespace_imputed(sample_raw_data: pd.DataFrame, tmp_path) -> None:
    """Verifies that TotalCharges whitespace ' ' for tenure 0 is coerced to 0.0."""
    p_path = tmp_path / "test_clean.parquet"
    c_path = tmp_path / "test_clean.csv"

    df_clean, audit = clean_data(sample_raw_data, output_parquet_path=p_path, output_csv_path=c_path)

    # In sample_raw_data, row 0 has tenure 0 and whitespace TotalCharges
    assert df_clean.loc[0, "TotalCharges"] == 0.0
    assert audit["totalcharges_whitespace_imputed"] == 1
    assert df_clean["TotalCharges"].dtype == "float64"
    assert p_path.exists()
    assert c_path.exists()


def test_clean_data_reconciliation(sample_raw_data: pd.DataFrame, tmp_path) -> None:
    """Verifies that input row count exactly matches output row count."""
    df_clean, audit = clean_data(
        sample_raw_data,
        output_parquet_path=tmp_path / "clean.parquet",
        output_csv_path=tmp_path / "clean.csv",
    )
    assert len(df_clean) == len(sample_raw_data)
    assert audit["reconciliation_match"] is True


def test_clean_data_tenure_band_derivation(sample_raw_data: pd.DataFrame, tmp_path) -> None:
    """Verifies that tenure bands are derived accurately for all records."""
    df_clean, _ = clean_data(
        sample_raw_data,
        output_parquet_path=tmp_path / "clean.parquet",
        output_csv_path=tmp_path / "clean.csv",
    )
    assert "tenure_band" in df_clean.columns
    # Row 0 has tenure 0 -> should be 0-6m (Onboarding)
    assert "0-6m (Onboarding)" in df_clean.loc[0, "tenure_band"]
