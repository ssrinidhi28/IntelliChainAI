"""Simple, reproducible demand-forecasting baselines."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def make_daily_series(df: pd.DataFrame, product_id: str | None = None) -> pd.DataFrame:
    subset = df if product_id is None else df[df["product_id"] == product_id]
    if subset.empty:
        raise ValueError(f"No rows found for product_id={product_id!r}")
    series = subset.groupby("order_date")["quantity"].sum().sort_index()
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
    return pd.DataFrame({"date": full_index, "demand": series.reindex(full_index, fill_value=0).values})

def feature_frame(series_df: pd.DataFrame) -> pd.DataFrame:
    out = series_df.copy().sort_values("date")
    for lag in [1, 7, 14, 28]:
        out[f"lag_{lag}"] = out["demand"].shift(lag)
    for window in [7, 14, 28]:
        out[f"rolling_mean_{window}"] = out["demand"].shift(1).rolling(window).mean()
    out["day_of_week"] = pd.to_datetime(out["date"]).dt.dayofweek
    out["day_of_year"] = pd.to_datetime(out["date"]).dt.dayofyear
    out["month"] = pd.to_datetime(out["date"]).dt.month
    return out.dropna().reset_index(drop=True)

def evaluate_forecast(df: pd.DataFrame, horizon: int = 28) -> tuple[pd.DataFrame, dict]:
    """Chronological holdout evaluation; features use past values only."""
    features = feature_frame(df)
    if len(features) <= horizon + 20:
        raise ValueError("Not enough daily history for this horizon; use more data or a smaller horizon.")
    split = len(features) - horizon
    train, test = features.iloc[:split], features.iloc[split:]
    cols = [c for c in features.columns if c not in {"date", "demand"}]
    X_train, y_train = train[cols], train["demand"]
    X_test, y_test = test[cols], test["demand"]
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=160, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=120, random_state=42),
    }
    results, preds = [], []
    baseline = test["lag_7"].to_numpy()
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = np.maximum(0, model.predict(X_test))
        results.append({"model": name, "mae": mean_absolute_error(y_test, pred),
                        "rmse": mean_squared_error(y_test, pred) ** 0.5})
        preds.append(pd.DataFrame({"date": test["date"].values, "actual": y_test.values,
                                   "predicted": pred, "model": name}))
    results.append({"model": "SeasonalNaiveLag7", "mae": mean_absolute_error(y_test, baseline),
                    "rmse": mean_squared_error(y_test, baseline) ** 0.5})
    preds.append(pd.DataFrame({"date": test["date"].values, "actual": y_test.values,
                               "predicted": baseline, "model": "SeasonalNaiveLag7"}))
    return pd.concat(preds, ignore_index=True), {
        "holdout_days": int(horizon),
        "metrics": sorted(results, key=lambda x: x["mae"]),
        "note": "Chronological holdout. Results depend on the dataset and are not production guarantees."
    }
