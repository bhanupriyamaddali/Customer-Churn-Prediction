"""
Model training with Optuna hyperparameter search.
Uses temporal cross-validation to avoid data leakage.
"""
import pandas as pd
import numpy as np
import yaml
import joblib
import os
import optuna
import warnings

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

import sys
sys.path.insert(0, os.path.dirname(__file__))
from features import get_feature_matrix, temporal_split
from evaluate import find_cost_threshold

CONFIG_PATH = "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "eval_metric": "aucpr",
        "random_state": 42,
    }
    model = XGBClassifier(**params, early_stopping_rounds=30)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    proba = model.predict_proba(X_val)[:, 1]
    # optimize average precision (better than AUC for imbalanced classes)
    return average_precision_score(y_val, proba)


def train(data_path=None, config_path=CONFIG_PATH):
    cfg = load_config()
    data_path = data_path or cfg["data"]["train_path"]

    print("Loading data...")
    df = pd.read_csv(data_path, parse_dates=[cfg["business"]["date_col"]])
    y_col = cfg["business"]["churn_col"]
    y = (df[y_col] == "Yes").astype(int)

    train_df, val_df, test_df = temporal_split(
        df,
        date_col=cfg["business"]["date_col"],
        train_months=cfg["model"]["train_months"],
        val_months=cfg["model"]["val_months"],
    )

    X_train, encoders = get_feature_matrix(train_df, fit_encoders=True)
    X_val, _ = get_feature_matrix(val_df, encoders=encoders, fit_encoders=False)
    X_test, _ = get_feature_matrix(test_df, encoders=encoders, fit_encoders=False)

    y_train = y.loc[train_df.index]
    y_val = y.loc[val_df.index]
    y_test = y.loc[test_df.index]

    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"Churn rate — train: {y_train.mean():.2%} | val: {y_val.mean():.2%}")

    print(f"\nRunning Optuna search ({cfg['model']['n_trials']} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=cfg["model"]["n_trials"],
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_params["scale_pos_weight"] = (y_train == 0).sum() / (y_train == 1).sum()
    best_params["random_state"] = 42

    print(f"\nBest trial AP: {study.best_value:.4f}")
    print(f"Best params: {best_params}")

    # refit on train+val with best params
    X_tv = pd.concat([X_train, X_val])
    y_tv = pd.concat([y_train, y_val])
    final_model = XGBClassifier(**best_params, early_stopping_rounds=30)
    final_model.fit(X_tv, y_tv, eval_set=[(X_test, y_test)], verbose=100)

    # find business-optimal threshold
    test_proba = final_model.predict_proba(X_test)[:, 1]
    threshold = find_cost_threshold(
        y_test, test_proba,
        fn_cost=cfg["model"]["threshold_fn_cost"]
    )

    os.makedirs(os.path.dirname(cfg["data"]["model_path"]), exist_ok=True)
    joblib.dump({
        "model": final_model,
        "encoders": encoders,
        "threshold": threshold,
        "best_params": best_params,
        "feature_names": list(X_train.columns),
    }, cfg["data"]["model_path"])

    print(f"\nModel saved → {cfg['data']['model_path']}")
    print(f"Decision threshold: {threshold:.3f}")
    return final_model, threshold


if __name__ == "__main__":
    train()
