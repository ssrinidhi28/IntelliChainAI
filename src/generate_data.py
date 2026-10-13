"""Generate deterministic synthetic supply-chain order data for demos."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CATEGORIES = ["Electronics", "Home", "Apparel", "Office", "Outdoor"]
REGIONS = ["North", "South", "East", "West", "Central"]
SEGMENTS = ["Consumer", "Small Business", "Enterprise"]

def generate_data(days: int = 540, seed: int = 42, n_products: int = 20,
                  orders_per_day: int = 24) -> pd.DataFrame:
    if days < 30:
        raise ValueError("days must be at least 30")
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize()
    dates = pd.date_range(end=end, periods=days, freq="D")
    rows = []
    product_ids = [f"SKU-{i:03d}" for i in range(1, n_products + 1)]
    category_by_product = {p: CATEGORIES[i % len(CATEGORIES)] for i, p in enumerate(product_ids)}
    base_price = {p: float(rng.uniform(8, 250)) for p in product_ids}
    for day_idx, day in enumerate(dates):
        n_orders = max(5, int(rng.poisson(orders_per_day)))
        seasonal = 1 + 0.20 * np.sin(2 * np.pi * day.dayofyear / 365.25)
        weekend_factor = 0.88 if day.dayofweek >= 5 else 1.0
        for j in range(n_orders):
            product = str(rng.choice(product_ids))
            qty = max(1, int(rng.negative_binomial(3, 0.45)))
            price = base_price[product] * rng.uniform(0.92, 1.08)
            cost = price * rng.uniform(0.48, 0.78)
            lead = int(rng.integers(2, 8))
            shipping = max(1, int(round(rng.normal(lead + 0.5, 2.0))))
            order_date = day
            ship_date = order_date + pd.Timedelta(days=max(0, shipping - int(rng.integers(1, 3))))
            delivery_date = order_date + pd.Timedelta(days=shipping)
            inventory = max(0, int(rng.normal(75 * seasonal * weekend_factor, 28)))
            rows.append({
                "order_id": f"ORD-{day.strftime('%Y%m%d')}-{j+1:04d}",
                "order_date": order_date,
                "ship_date": ship_date,
                "delivery_date": delivery_date,
                "product_id": product,
                "category": category_by_product[product],
                "region": str(rng.choice(REGIONS)),
                "quantity": qty,
                "unit_price": round(price, 2),
                "unit_cost": round(cost, 2),
                "inventory_level": inventory,
                "supplier_lead_days": lead,
                "shipping_days": shipping,
                "is_late": int(shipping > lead),
                "customer_segment": str(rng.choice(SEGMENTS, p=[0.68, 0.23, 0.09])),
            })
    df = pd.DataFrame(rows)
    # Add mild, realistic imperfections so cleaning steps can be demonstrated.
    if len(df) > 100:
        missing_idx = rng.choice(df.index, size=max(1, len(df) // 500), replace=False)
        df.loc[missing_idx, "region"] = np.nan
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/supply_chain_demo.csv")
    parser.add_argument("--days", type=int, default=540)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--products", type=int, default=20)
    parser.add_argument("--orders-per-day", type=int, default=24)
    args = parser.parse_args()
    df = generate_data(args.days, args.seed, args.products, args.orders_per_day)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df):,} rows to {path}")

if __name__ == "__main__":
    main()
