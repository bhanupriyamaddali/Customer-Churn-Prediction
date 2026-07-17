"""
Synthetic customer data generator for local dev and testing.
Production version connects to Snowflake feature store.

Run: python src/generate_data.py
Output: data/churn_train.csv, data/new_customers.csv
"""
import pandas as pd
import numpy as np
import os

SEED = 42
N = 7000


def generate(n=N, seed=SEED):
    rng = np.random.default_rng(seed)

    contracts    = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    payment      = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n)
    internet     = rng.choice(["Fiber optic", "DSL", "No"], n, p=[0.44, 0.34, 0.22])
    tech_support = rng.choice(["Yes", "No", "No internet service"], n)
    online_sec   = rng.choice(["Yes", "No", "No internet service"], n)
    device_prot  = rng.choice(["Yes", "No", "No internet service"], n)

    tenure            = rng.integers(1, 73, n)
    monthly_charges   = rng.uniform(20, 120, n).round(2)
    total_charges     = (monthly_charges * tenure * rng.uniform(0.85, 1.0, n)).round(2)
    support_calls     = rng.poisson(1.5, n)
    days_since_login  = rng.integers(0, 60, n)
    product_changes   = rng.poisson(0.8, n)

    # churn probability — higher for month-to-month, fiber, high charges, new customers
    churn_logit = (
        -2.5
        + 1.2  * (contracts == "Month-to-month")
        + 0.8  * (internet == "Fiber optic")
        + 0.6  * (monthly_charges > 80)
        + 0.9  * (tenure < 12)
        + 0.5  * (support_calls >= 3)
        - 0.7  * (tech_support == "Yes")
        - 0.5  * (online_sec == "Yes")
        + rng.normal(0, 0.5, n)
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn_flag = (rng.random(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customerID":       [f"CUST-{i:06d}" for i in range(n)],
        "tenure":           tenure,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,
        "Contract":         contracts,
        "PaymentMethod":    payment,
        "InternetService":  internet,
        "TechSupport":      tech_support,
        "OnlineSecurity":   online_sec,
        "DeviceProtection": device_prot,
        "NumSupportCalls":  support_calls,
        "NumProductChanges": product_changes,
        "DaysSinceLastLogin": days_since_login,
        "Churn":            np.where(churn_flag, "Yes", "No"),
        "signup_date":      pd.date_range("2021-01-01", periods=n, freq="6H"),
    })

    os.makedirs("data", exist_ok=True)
    train = df.sample(frac=0.85, random_state=seed)
    new   = df.drop(train.index)

    train.to_csv("data/churn_train.csv", index=False)
    new.drop(columns=["Churn"]).to_csv("data/new_customers.csv", index=False)

    churn_rate = churn_flag.mean()
    print(f"Generated {n:,} customers | Churn rate: {churn_rate:.1%}")
    print(f"  data/churn_train.csv    ({len(train):,} rows)")
    print(f"  data/new_customers.csv  ({len(new):,} rows, no label)")
    return df


if __name__ == "__main__":
    generate()
