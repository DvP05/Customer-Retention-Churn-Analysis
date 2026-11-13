"""
Unit tests for canonical metrics: churn prevalence, retention complement, and exposure proxy.
"""

import pandas as pd

from retention.analysis.metrics import (
    calculate_headline_kpis,
    churn_prevalence,
    monthly_charge_exposure_proxy,
    retention_complement,
)


def test_churn_prevalence_calculation(sample_raw_data: pd.DataFrame) -> None:
    """Verifies churn prevalence on known 10-row dataset with 3 churned."""
    res = churn_prevalence(sample_raw_data)
    assert res["labeled_count"] == 10
    assert res["churned_count"] == 3
    assert res["prevalence"] == 0.30
    assert res["prevalence_pct"] == 30.0


def test_retention_complement_calculation(sample_raw_data: pd.DataFrame) -> None:
    """Verifies retention complement is 1 - churn prevalence."""
    res = retention_complement(sample_raw_data)
    assert res["retained_count"] == 7
    assert res["retention_complement"] == 0.70
    assert res["retention_complement_pct"] == 70.0


def test_monthly_charge_exposure_proxy(sample_raw_data: pd.DataFrame) -> None:
    """Verifies sum of MonthlyCharges for churned customers."""
    # Churned rows in sample: index 1 (89.10), index 4 (105.00), index 6 (75.30)
    # Total churned exposure = 89.10 + 105.00 + 75.30 = 269.40
    res = monthly_charge_exposure_proxy(sample_raw_data)
    assert res["churn_exposure_proxy"] == 269.40


def test_zero_division_protection() -> None:
    """Verifies that empty DataFrame does not raise ZeroDivisionError."""
    empty_df = pd.DataFrame(columns=["Churn", "MonthlyCharges", "tenure"])
    prev = churn_prevalence(empty_df)
    assert prev["prevalence"] == 0.0
    assert prev["churned_count"] == 0

    kpis = calculate_headline_kpis(empty_df)
    assert kpis["total_customers"] == 0
    assert kpis["churn_prevalence"] == 0.0
