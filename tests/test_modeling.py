"""
Unit tests for model pipeline fitting, predictions, and determinism.
"""

import numpy as np
import pandas as pd

from retention.data.clean import clean_data
from retention.features.build import extract_features_and_target
from retention.modeling.train import create_model_pipeline


def test_model_pipeline_fit_predict(sample_raw_data: pd.DataFrame, tmp_path) -> None:
    """Verifies that full pipeline fits on synthetic data and outputs valid probabilities in [0, 1]."""
    df_clean, _ = clean_data(
        sample_raw_data,
        output_parquet_path=tmp_path / "clean.parquet",
        output_csv_path=tmp_path / "clean.csv",
    )
    X, y = extract_features_and_target(df_clean)

    pipe = create_model_pipeline("logistic_regression")
    pipe.fit(X, y)

    probs = pipe.predict_proba(X)[:, 1]
    assert len(probs) == len(X)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_model_determinism(sample_raw_data: pd.DataFrame, tmp_path) -> None:
    """Verifies that model predictions are deterministic across fits with fixed seed."""
    df_clean, _ = clean_data(
        sample_raw_data,
        output_parquet_path=tmp_path / "clean.parquet",
        output_csv_path=tmp_path / "clean.csv",
    )
    X, y = extract_features_and_target(df_clean)

    pipe1 = create_model_pipeline("logistic_regression")
    pipe1.fit(X, y)
    probs1 = pipe1.predict_proba(X)[:, 1]

    pipe2 = create_model_pipeline("logistic_regression")
    pipe2.fit(X, y)
    probs2 = pipe2.predict_proba(X)[:, 1]

    np.testing.assert_allclose(probs1, probs2, atol=1e-7)
