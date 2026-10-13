"""Delivery-risk classifier and held-out evaluation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "is_late"
# Exclude shipping_days: it directly determines the synthetic target and would leak the answer.
NUMERIC = ["quantity", "unit_price", "unit_cost", "inventory_level", "supplier_lead_days"]
CATEGORICAL = ["category", "region", "product_id", "customer_segment"]

def train_delivery_risk(df: pd.DataFrame):
    usable = df.dropna(subset=[TARGET]).copy()
    features = [c for c in NUMERIC + CATEGORICAL if c in usable.columns]
    X, y = usable[features], usable[TARGET].astype(int)
    if y.nunique() < 2:
        raise ValueError("Delivery-risk target must contain both classes.")
    numeric = [c for c in NUMERIC if c in features]
    categorical = [c for c in CATEGORICAL if c in features]
    transformer = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = Pipeline([
        ("preprocess", transformer),
        ("classifier", RandomForestClassifier(n_estimators=180, min_samples_leaf=3,
                                              class_weight="balanced", random_state=42, n_jobs=-1)),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "test_rows": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
    except ValueError:
        metrics["roc_auc"] = None
    # Permutation importance operates on original input columns through the pipeline.
    importance = permutation_importance(model, X_test, y_test, n_repeats=5,
                                        random_state=42, scoring="f1", n_jobs=-1)
    imp_df = pd.DataFrame({"feature": features, "importance_mean": importance.importances_mean,
                           "importance_std": importance.importances_std}).sort_values(
                               "importance_mean", ascending=False)
    return model, metrics, imp_df
