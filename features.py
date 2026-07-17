"""
Batch scoring pipeline. Outputs customer-level churn scores + revenue at risk.
In production this runs nightly, results land in Snowflake for the Streamlit dashboard.
"""
import pandas as pd
import numpy as np
import joblib
import argparse
import yaml
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from features import get_feature_matrix


def score(input_path, output_path, model_path="models/churn_model.pkl", config_path="config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    bundle = joblib.load(model_path)
    model = bundle["model"]
    encoders = bundle["encoders"]
    threshold = bundle["threshold"]

    df = pd.read_csv(input_path)
    X, _ = get_feature_matrix(df, encoders=encoders, fit_encoders=False)

    df["churn_prob"] = model.predict_proba(X)[:, 1]
    df["churn_flag"] = (df["churn_prob"] >= threshold).astype(int)

    rev_col = cfg["business"]["revenue_col"]
    if rev_col in df.columns:
        df["expected_mrr_loss"] = df["churn_prob"] * df[rev_col]
    else:
        df["expected_mrr_loss"] = np.nan

    df = df.sort_values("churn_prob", ascending=False)
    df.to_csv(output_path, index=False)

    n_flagged = df["churn_flag"].sum()
    total_mrr = df.loc[df["churn_flag"] == 1, "expected_mrr_loss"].sum()

    print(f"Scored {len(df):,} customers → {output_path}")
    print(f"At-risk customers (threshold={threshold:.2f}): {n_flagged:,}")
    if not np.isnan(total_mrr):
        print(f"Total MRR at risk: ${total_mrr:,.0f}/month")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model",  default="models/churn_model.pkl")
    args = parser.parse_args()
    score(args.input, args.output, model_path=args.model)
