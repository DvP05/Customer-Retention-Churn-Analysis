"""
Pytest configuration and synthetic fixtures for unit tests.
Creates lightweight, deterministic DataFrames without requiring full raw dataset.
"""

import pandas as pd
import pytest


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Returns a minimal 10-row synthetic raw customer DataFrame with all 21 columns."""
    data = {
        "customerID": [f"CUST-{i:03d}" for i in range(1, 11)],
        "gender": ["Male", "Female"] * 5,
        "SeniorCitizen": [0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
        "Partner": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
        "Dependents": ["No", "Yes", "No", "No", "Yes", "No", "No", "Yes", "No", "No"],
        "tenure": [0, 5, 12, 24, 48, 60, 2, 72, 1, 36],
        "PhoneService": ["Yes"] * 10,
        "MultipleLines": ["No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "No", "DSL", "Fiber optic", "DSL", "Fiber optic", "No", "DSL", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes", "No internet service", "No", "Yes", "No", "No", "No internet service", "Yes", "No"],
        "OnlineBackup": ["Yes", "No", "No internet service", "Yes", "No", "Yes", "No", "No internet service", "No", "Yes"],
        "DeviceProtection": ["No", "Yes", "No internet service", "No", "Yes", "No", "Yes", "No internet service", "No", "No"],
        "TechSupport": ["No", "No", "No internet service", "Yes", "No", "Yes", "No", "No internet service", "No", "Yes"],
        "StreamingTV": ["No", "Yes", "No internet service", "No", "Yes", "No", "Yes", "No internet service", "No", "Yes"],
        "StreamingMovies": ["No", "Yes", "No internet service", "Yes", "No", "No", "Yes", "No internet service", "No", "Yes"],
        "Contract": ["Month-to-month", "Month-to-month", "One year", "Two year", "Month-to-month", "Two year", "Month-to-month", "Two year", "Month-to-month", "One year"],
        "PaperlessBilling": ["Yes", "Yes", "No", "No", "Yes", "No", "Yes", "No", "Yes", "Yes"],
        "PaymentMethod": ["Electronic check", "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"] * 2,
        "MonthlyCharges": [29.85, 89.10, 19.95, 55.20, 105.00, 65.40, 75.30, 20.00, 45.00, 95.50],
        # Row 1 has tenure 0 and whitespace string " "
        "TotalCharges": [" ", "445.50", "239.40", "1324.80", "5040.00", "3924.00", "150.60", "1440.00", "45.00", "3438.00"],
        # 3 Churned ("Yes"), 7 Retained ("No") -> Churn Prevalence = 30.0%
        "Churn": ["No", "Yes", "No", "No", "Yes", "No", "Yes", "No", "No", "No"],
    }
    return pd.DataFrame(data)
