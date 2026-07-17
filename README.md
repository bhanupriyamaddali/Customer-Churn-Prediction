# Customer Churn Prediction & Revenue Impact Dashboard

Predicts which customers are most likely to churn in the next 30 days and quantifies the revenue at risk per segment. Built for a subscription business context — the model feeds a Streamlit dashboard used by CX and finance leadership to prioritize retention outreach.

---

## Problem

Most churn models optimize for AUC and stop there. This one was built around a specific business constraint: the retention team can only reach out to ~500 customers per week. That changes the problem from "maximize recall" to "rank the right 500 customers" — which means precision in the top decile matters more than overall accuracy.

The secondary output — revenue at risk — required mapping churn probability to expected monthly revenue loss at the customer segment level, not just flagging individual accounts.

---

## Approach

**Feature engineering first.** Raw features like `monthly_charges` and `tenure` are weak in isolation. What matters is the *relationship* between them — customers who pay a lot but haven't been around long are a very different risk profile than long-tenure customers with the same bill. Built ~20 engineered features including charge-per-tenure ratios, support call velocity, and contract proximity to renewal.

**Temporal validation.** Used a rolling-origin cross-validation scheme rather than a random train/test split. Churn prediction is a time-series problem — a random split leaks future information. Trained on months 1-18, validated on 19-21, tested on 22-24.

**Model selection.** Benchmarked logistic regression, random forest, XGBoost, and LightGBM. XGBoost won on the metric that mattered: precision@500 (top 500 riskiest customers). Final model: XGBoost with Optuna-tuned hyperparameters, class-weight adjusted for ~15% churn rate.

**Threshold calibration.** Default 0.5 threshold is wrong for imbalanced classes. Used a cost-sensitive threshold search: false negatives (missed churners) were estimated at 12x more costly than false positives (wasted outreach). Optimal threshold: 0.38.

**Explainability.** SHAP TreeExplainer provides both global feature importance and individual prediction explanations — important for the retention team to understand *why* a specific customer is flagged.

---

## Results

| Metric | Value | Baseline (majority class) |
|--------|-------|--------------------------|
| AUC-ROC | 0.89 | 0.50 |
| Precision@500 | 74% | ~15% |
| Recall (churners) | 82% | 0% |
| Revenue coverage | 68% of at-risk MRR in top 500 | — |

Precision@500 means 74% of the top 500 flagged customers actually churned — vs. a ~15% hit rate from random selection.

---

## Technical Highlights

- **Temporal cross-validation** — prevents data leakage, produces realistic estimates of production performance
- **Cost-sensitive threshold optimization** — threshold chosen using a business cost ratio (FN cost / FP cost), not arbitrary 0.5 cutoff
- **SHAP-based explanations** — each customer's top 3 churn drivers surfaced in the dashboard, not just a probability score
- **Revenue at risk calculation** — churn probability × MRR, aggregated by segment and tenure band for financial planning

---

## Stack

Python, XGBoost, LightGBM, SHAP, Optuna, scikit-learn, Pandas, NumPy, Streamlit, Matplotlib

---

## Project Structure

```
customer-churn-prediction/
├── src/
│   ├── features.py          # feature engineering + temporal split logic
│   ├── train.py             # model training with Optuna tuning
│   ├── evaluate.py          # temporal CV, precision@K, cost-sensitive threshold
│   ├── explain.py           # SHAP global + per-customer explanations
│   └── score.py             # batch scoring + revenue at risk calculation
├── notebooks/
│   └── 01_eda_and_feature_dev.ipynb
├── app.py                   # Streamlit dashboard
├── config.yaml              # model hyperparameters and business config
├── requirements.txt
└── .gitignore
```

---

## Quickstart

```bash
git clone https://github.com/bhanupriyamaddali/customer-churn-prediction
cd customer-churn-prediction
pip install -r requirements.txt

# train model (expects data/churn.csv — see config.yaml for schema)
python src/train.py

# score new customers
python src/score.py --input data/new_customers.csv --output data/scored.csv

# launch dashboard
streamlit run app.py
```

---

## Configuration

Key settings in `config.yaml`:

```yaml
model:
  n_trials: 50           # Optuna tuning trials
  cv_folds: 5
  threshold_fn_cost: 12  # cost ratio for threshold calibration

business:
  outreach_capacity: 500  # weekly retention team capacity
  revenue_col: MonthlyCharges
```
