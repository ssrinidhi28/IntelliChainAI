import pandas as pd
from src.data_utils import clean_orders, analytics_summary
from src.generate_data import generate_data

def test_generator_schema_and_nonempty():
    df = generate_data(days=35, seed=1, n_products=5, orders_per_day=8)
    assert len(df) > 0
    assert {"order_id", "order_date", "product_id", "quantity", "is_late"}.issubset(df.columns)

def test_cleaning_deduplicates_and_calculates_revenue():
    df = pd.DataFrame({
        "order_id": ["a", "a"], "order_date": ["2026-01-01", "2026-01-01"],
        "product_id": ["SKU-1", "SKU-1"], "category": ["Home", "Home"],
        "region": ["North", "North"], "quantity": [2, 2], "unit_price": [10, 10],
        "unit_cost": [4, 4], "inventory_level": [5, 5], "supplier_lead_days": [3, 3],
        "shipping_days": [4, 4], "is_late": [1, 1]
    })
    clean = clean_orders(df)
    assert len(clean) == 1
    assert clean.iloc[0]["revenue"] == 20
    assert clean.iloc[0]["estimated_margin"] == 12

def test_summary_counts():
    df = clean_orders(generate_data(days=35, seed=3, n_products=5, orders_per_day=8))
    summary = analytics_summary(df)
    assert summary["orders"] > 0
    assert summary["revenue"] >= 0
