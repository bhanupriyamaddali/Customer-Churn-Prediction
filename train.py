"""
Evaluation utilities for churn model.

Key metrics for this use case:
- Precision@K: what fraction of the top-K flagged customers actually churn?
  (More useful than raw AUC when the retention team has a fixed capacity)
- Cost-sensitive threshold: threshold chosen to minimize expected misclassification cost
- Revenue coverage: % of true churn MRR captured in top-K predictions
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, precision_recall_curve,
)
import matplotlib.pyplot as plt


def precision_at_k(y_true, y_score, k):
    """Precision of the top-K highest-scoring predictions."""
    top_k_idx = np.argsort(y_score)[::-1][:k]
    return y_true.iloc[top_k_idx].mean() if hasattr(y_true, "iloc") else y_true[top_k_idx].mean()


def recall_at_k(y_true, y_score, k):
    """Fraction of all actual churners captured in top-K predictions."""
    top_k_idx = np.argsort(y_score)[::-1][:k]
    hits = (y_true.iloc[top_k_idx] == 1).sum() if hasattr(y_true, "iloc") else (y_true[top_k_idx] == 1).sum()
    return hits / max(y_true.sum(), 1)


def revenue_coverage_at_k(y_true, y_score, revenue, k):
    """
    Fraction of total at-risk MRR captured in top-K predictions.
    Requires a revenue array aligned with y_true and y_score.
    """
    y_true = np.asarray(y_true)
    top_k_idx = np.argsort(y_score)[::-1][:k]
    churn_rev = revenue[y_true == 1].sum()
    captured_rev = revenue[top_k_idx][y_true[top_k_idx] == 1].sum()
    return captured_rev / max(churn_rev, 1)


def find_cost_threshold(y_true, proba, fn_cost=12, fp_cost=1):
    """
    Find the decision threshold that minimizes expected cost.
    fn_cost: cost of missing a churner (lost revenue + acquisition cost)
    fp_cost: cost of a false positive (wasted retention outreach)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    churn_rate = y_true.mean()

    # expected cost at each threshold
    fn_rate = 1 - recall
    fp_rate = precision  # approximation
    costs = fn_cost * fn_rate * churn_rate + fp_cost * (1 - precision) * (1 - churn_rate)
    best_idx = np.argmin(costs[:-1])  # last element has no threshold
    return float(thresholds[best_idx])


def full_report(y_true, y_score, capacity=500, revenue=None):
    """Print a complete model evaluation report."""
    threshold = find_cost_threshold(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    p_at_k = precision_at_k(y_true, y_score, k=capacity)
    r_at_k = recall_at_k(y_true, y_score, k=capacity)

    print("=" * 50)
    print("CHURN MODEL EVALUATION REPORT")
    print("=" * 50)
    print(f"AUC-ROC:              {auc:.4f}")
    print(f"Average Precision:    {ap:.4f}")
    print(f"Precision@{capacity}:         {p_at_k:.4f}")
    print(f"Recall@{capacity}:            {r_at_k:.4f}")
    print(f"Decision threshold:   {threshold:.3f}")
    if revenue is not None:
        rev_cov = revenue_coverage_at_k(y_true, y_score, np.asarray(revenue), k=capacity)
        print(f"Revenue coverage@{capacity}:  {rev_cov:.2%}")
    print()
    print("Classification report (at cost-optimal threshold):")
    print(classification_report(y_true, y_pred, target_names=["retained", "churned"]))


def plot_precision_recall(y_true, y_score, capacity=500, save_path=None):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    baseline = y_true.mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(recall, precision, lw=2, color="#2c3e50")
    axes[0].axhline(baseline, linestyle="--", color="gray", label=f"Random baseline ({baseline:.2%})")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall Curve")
    axes[0].legend()

    ks = range(100, min(2001, len(y_true)), 100)
    p_at_ks = [precision_at_k(y_true, y_score, k) for k in ks]
    axes[1].plot(list(ks), p_at_ks, lw=2, color="#c0392b")
    axes[1].axvline(capacity, linestyle="--", color="gray", label=f"Team capacity ({capacity})")
    axes[1].set_xlabel("Top-K predictions")
    axes[1].set_ylabel("Precision@K")
    axes[1].set_title("Precision at K")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
