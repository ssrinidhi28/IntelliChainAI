"""Run data validation, analytics, forecasting, and delivery-risk modelling."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
from src.data_utils import load_orders, analytics_summary, daily_demand
from src.forecasting import make_daily_series, evaluate_forecast
from src.risk_model import train_delivery_risk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/supply_chain_demo.csv")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--product", default="SKU-001")
    parser.add_argument("--horizon", type=int, default=28)
    args = parser.parse_args()
    report_dir, model_dir = Path(args.output_dir), Path(args.model_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_orders(args.input)
    summary = analytics_summary(df)
    (report_dir / "analytics_summary.json").write_text(json.dumps(summary, indent=2))
    daily_demand(df).to_csv(report_dir / "daily_demand.csv", index=False)

    product = args.product if args.product in set(df["product_id"]) else str(df["product_id"].mode().iloc[0])
    series = make_daily_series(df, product)
    forecast_df, forecast_metrics = evaluate_forecast(series, horizon=args.horizon)
    forecast_df.to_csv(report_dir / "forecast_results.csv", index=False)
    (report_dir / "forecast_metrics.json").write_text(json.dumps(forecast_metrics, indent=2))

    model, risk_metrics, importance = train_delivery_risk(df)
    joblib.dump(model, model_dir / "delivery_risk_model.joblib")
    importance.to_csv(report_dir / "delivery_risk_feature_importance.csv", index=False)
    (report_dir / "delivery_risk_metrics.json").write_text(json.dumps(risk_metrics, indent=2))

    print("Pipeline completed.")
    print(f"Rows analysed: {len(df):,}")
    print(f"Analytics summary: {report_dir / 'analytics_summary.json'}")
    print(f"Forecast metrics: {report_dir / 'forecast_metrics.json'}")
    print(f"Delivery-risk metrics: {report_dir / 'delivery_risk_metrics.json'}")
    print(f"Selected forecast product: {product}")

if __name__ == "__main__":
    main()
