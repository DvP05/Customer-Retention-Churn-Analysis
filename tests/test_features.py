"""
Unit tests for feature extraction and leakage prevention.
"""

import pandas as pd

from retention.config import config
from retention.data.clean import clean_data
from retention.features.build import build_preprocessor, extract_features_and_target


def test_customer_id_strictly_excluded(sample_raw_data: pd.DataFrame, tmp_path) -> None:
    """Verifies that customerID is never included in the predictor matrix X."""
    df_clean, _ = clean_data(
        sample_raw_data,
        output_parquet_path=tmp_path / "clean.parquet",
        output_csv_path=tmp_path / "clean.csv",
    )
    X, y = extract_features_and_target(df_clean)

    assert config.id_column not in X.columns
    assert "Churn" not in X.columns
    assert "churn_binary" not in X.columns
    assert len(y) == len(sample_raw_data)


def test_preprocessor_transformation(sample_raw_data: pd.DataFrame, tmp_path) -> None:
    """Verifies that ColumnTransformer correctly encodes and scales features without NaN."""
    df_clean, _ = clean_data(
        sample_raw_data,
        output_parquet_path=tmp_path / "clean.parquet",
        output_csv_path=tmp_path / "clean.csv",
    )
    X, _ = extract_features_and_target(df_clean)

    num_cols = list(config.numeric_columns)
    cat_cols = list(config.categorical_columns)

    preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=True)
    X_trans = preprocessor.fit_transform(X)

    assert X_trans.shape[0] == len(sample_raw_data)
    assert not pd.isna(X_trans).any()
