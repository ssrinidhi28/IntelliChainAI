"""Data loading, schema validation, cleaning, and business KPI helpers."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {
    "order_id", "order_date", "product_id", "category", "region", "quantity",
    "unit_price", "unit_cost", "inventory_level", "supplier_lead_days",
    "shipping_days", "is_late"
}
DATE_COLUMNS = ["order_date", "ship_date", "delivery_date"]

def load_orders(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")
    for col in DATE_COLUMNS:
        if col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return clean_orders(df)

def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy; never mutate the caller's dataframe."""
    clean = df.copy()
    if "order_id" in clean:
        clean = clean.drop_duplicates(subset=["order_id"], keep="first")
    for col in DATE_COLUMNS:
        if col in clean:
            clean[col] = pd.to_datetime(clean[col], errors="coerce")
    numeric = ["quantity", "unit_price", "unit_cost", "inventory_level",
               "supplier_lead_days", "shipping_days", "is_late"]
    for col in numeric:
        if col in clean:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")
    for col in ["category", "region", "product_id", "customer_segment"]:
        if col in clean:
            clean[col] = clean[col].fillna("Unknown").astype(str)
    for col in ["quantity", "unit_price", "unit_cost", "inventory_level",
                "supplier_lead_days", "shipping_days"]:
        if col in clean:
            clean[col] = clean[col].fillna(clean[col].median())
    if "is_late" in clean:
        clean["is_late"] = clean["is_late"].fillna(0).astype(int).clip(0, 1)
    if "quantity" in clean:
        clean = clean[clean["quantity"] >= 0]
    if "unit_price" in clean:
        clean = clean[clean["unit_price"] >= 0]
    if "order_date" in clean:
        clean = clean.dropna(subset=["order_date"])
    if {"unit_price", "quantity"}.issubset(clean.columns):
        clean["revenue"] = clean["unit_price"] * clean["quantity"]
    if {"unit_cost", "quantity"}.issubset(clean.columns):
        clean["estimated_cost"] = clean["unit_cost"] * clean["quantity"]
    if {"revenue", "estimated_cost"}.issubset(clean.columns):
        clean["estimated_margin"] = clean["revenue"] - clean["estimated_cost"]
    return clean.reset_index(drop=True)

def analytics_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"orders": 0, "revenue": 0.0, "estimated_margin": 0.0,
                "late_delivery_rate": 0.0, "unique_products": 0}
    revenue = df.get("revenue", df["unit_price"] * df["quantity"]).sum()
    margin = df.get("estimated_margin", revenue - (df["unit_cost"] * df["quantity"]).sum()).sum()
    return {
        "orders": int(df["order_id"].nunique()) if "order_id" in df else int(len(df)),
        "units_sold": int(df["quantity"].sum()),
        "revenue": round(float(revenue), 2),
        "estimated_margin": round(float(margin), 2),
        "late_delivery_rate": round(float(df["is_late"].mean()), 4),
        "unique_products": int(df["product_id"].nunique()),
        "date_min": str(df["order_date"].min().date()),
        "date_max": str(df["order_date"].max().date()),
    }

def daily_demand(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby(["order_date", "product_id"], as_index=False)["quantity"].sum()
    return out.rename(columns={"order_date": "date", "quantity": "demand"}).sort_values(["product_id", "date"])
