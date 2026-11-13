"""
Unit tests for cohort capabilities and tenure band non-overlapping partitions.
"""

import pandas as pd

from retention.analysis.cohorts import analyze_tenure_bands, check_cohort_capability
from retention.config import config


def test_cohort_capability_notice_on_snapshot(sample_raw_data: pd.DataFrame) -> None:
    """Verifies that static snapshot is flagged as lacking dated events."""
    supported, notice = check_cohort_capability(sample_raw_data)
    assert supported is False
    assert "Static Snapshot Notice" in notice


def test_tenure_band_aggregation(sample_raw_data: pd.DataFrame) -> None:
    """Verifies that analyze_tenure_bands groups counts and exposures cleanly."""
    # Pre-add tenure band
    bins = config.tenure_bins
    labels = config.tenure_labels
    df = sample_raw_data.copy()
    df["tenure_band"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=True, include_lowest=True).astype(str)

    band_df = analyze_tenure_bands(df)
    assert not band_df.empty
    assert band_df["customer_count"].sum() == len(sample_raw_data)
