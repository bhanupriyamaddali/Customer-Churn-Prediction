"""
Unit tests for churn feature engineering.
Run: pytest tests/test_features.py -v
"""
import pandas as pd
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from features import engineer, get_feature_matrix


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "customerID":       ["C001", "C002", "C003"],
        "tenure":           [2, 24, 0],
        "MonthlyCharges":   [70.0, 50.0, 30.0],
        "TotalCharges":     [140.0, 1200.0, 0.0],
        "Contract":         ["Month-to-month", "Two year", "Month-to-month"],
        "PaymentMethod":    ["Electronic check", "Bank transfer", "Mailed check"],
        "InternetService":  ["Fiber optic", "DSL", "No"],
        "TechSupport":      ["No", "Yes", "No internet service"],
        "OnlineSecurity":   ["No", "Yes", "No internet service"],
        "DeviceProtection": ["No", "No", "No internet service"],
        "NumSupportCalls":  [0, 1, 3],
        "NumProductChanges":[0, 2, 0],
        "DaysSinceLastLogin":[5, 30, 2],
        "Churn":            ["No", "No", "Yes"],
        "signup_date":      pd.to_datetime(["2023-01-01", "2022-01-01", "2023-10-01"]),
    })


def test_engineer_creates_expected_columns(sample_df):
    out = engineer(sample_df)
    assert "avg_monthly_charge" in out.columns
    assert "tenure_band" in out.columns
    assert "support_call_rate" in out.columns
    assert "near_renewal" in out.columns
    assert "service_count" in out.columns


def test_avg_monthly_charge_calculation(sample_df):
    out = engineer(sample_df)
    # avg charge = TotalCharges / (tenure + 1)
    expected = 140.0 / (2 + 1)
    assert abs(out.loc[0, "avg_monthly_charge"] - expected) < 0.01


def test_zero_tenure_no_division_error(sample_df):
    """tenure=0 should not cause division by zero."""
    out = engineer(sample_df)
    assert not out["avg_monthly_charge"].isna().any()
    assert not out["support_call_rate"].isna().any()


def test_tenure_band_labels(sample_df):
    out = engineer(sample_df)
    # tenure=2 should be "0-6mo", tenure=24 should be "1-2yr"
    assert str(out.loc[0, "tenure_band"]) == "0-6mo"
    assert str(out.loc[1, "tenure_band"]) == "1-2yr"


def test_support_call_rate_ordering(sample_df):
    """Customer with more support calls should have higher rate, controlling for tenure."""
    out = engineer(sample_df)
    # C003: 3 calls, tenure=0 → high rate; C001: 0 calls, tenure=2 → low rate
    assert out.loc[2, "support_call_rate"] > out.loc[0, "support_call_rate"]


def test_get_feature_matrix_no_label_column(sample_df):
    """Feature matrix should not include the target column."""
    X, encoders = get_feature_matrix(sample_df, fit_encoders=True)
    assert "Churn" not in X.columns
    assert "churn_binary" not in X.columns


def test_get_feature_matrix_no_nulls(sample_df):
    X, _ = get_feature_matrix(sample_df, fit_encoders=True)
    assert X.isna().sum().sum() == 0


def test_encoder_reuse_on_new_data(sample_df):
    """Encoders from training should work on new data without refitting."""
    X_train, encoders = get_feature_matrix(sample_df, fit_encoders=True)
    new_df = sample_df.copy()
    X_new, _ = get_feature_matrix(new_df, encoders=encoders, fit_encoders=False)
    assert X_train.shape[1] == X_new.shape[1]
