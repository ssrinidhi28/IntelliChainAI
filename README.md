# IntelliChain AI
### End-to-end predictive analytics, machine learning, and deep learning for supply-chain optimization

IntelliChain AI is a portfolio-ready reference project that turns retail order and inventory data into business insights and predictive signals. It combines data preparation, exploratory analytics, forecasting baselines, delivery-risk classification, optional deep-learning experiments, explainability, and an interactive Streamlit dashboard.

> **Important:** The repository includes a synthetic-data generator so it runs without private or licensed datasets. The generated data is for software demonstration only; metrics from it are not evidence of real-world performance. Replace it with a suitable public or approved dataset before making claims about business impact.

## What it demonstrates

- **Data analytics:** data quality checks, exploratory analysis, business KPIs, and CSV exports.
- **Machine learning:** demand forecasting with scikit-learn models and late-delivery risk classification.
- **Deep learning:** optional PyTorch LSTM time-series model, with an informative fallback when PyTorch is not installed.
- **Explainability:** permutation feature importance for the delivery-risk model.
- **Deployment:** Streamlit dashboard for exploring orders, trends, forecasts, and model outputs.
- **Reproducibility:** configuration, seeded synthetic data, modular source files, and tests.

## Architecture

```text
Synthetic or approved input data
           |
           v
   Validation + cleaning
           |
           +--------------------+
           |                    |
           v                    v
  Analytics / KPI report   Feature engineering
                                |
                   +------------+------------+
                   |                         |
                   v                         v
            Demand forecasting       Delivery-risk ML
                   |                         |
                   +------------+------------+
                                |
                                v
                  Streamlit decision dashboard
```

## Quick start

Python 3.10+ is recommended.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate reproducible demo data
python -m src.generate_data --output data/supply_chain_demo.csv --days 540 --seed 42

# 4. Run the analytics and ML pipeline
python -m src.train_pipeline --input data/supply_chain_demo.csv --output-dir reports

# 5. Launch the dashboard
streamlit run app.py
```

The dashboard looks for `data/supply_chain_demo.csv`. If it is absent, it creates demo data automatically.

## Optional deep-learning experiment

Install the optional dependency:

```bash
pip install torch
python -m src.deep_learning --input data/supply_chain_demo.csv --target-product SKU-001
```

The script trains an LSTM on one product's daily demand and reports MAE/RMSE on a chronological holdout. This is an educational baseline, not a production forecasting system. The pipeline never silently substitutes a fabricated deep-learning score.

## Input data schema

The included generator produces these fields:

| Column | Meaning |
|---|---|
| `order_id` | Unique order identifier |
| `order_date` | Order date |
| `ship_date` | Shipment date |
| `delivery_date` | Delivery date |
| `product_id` | SKU identifier |
| `category` | Product category |
| `region` | Sales region |
| `quantity` | Units ordered |
| `unit_price` | Unit price |
| `unit_cost` | Estimated unit cost |
| `inventory_level` | Inventory available at order time |
| `supplier_lead_days` | Expected supplier lead time |
| `shipping_days` | Actual shipping duration |
| `is_late` | Whether delivery exceeded the expected shipping duration |
| `customer_segment` | Synthetic customer segment |

For a real dataset, map its columns to this schema or adapt `src/data_utils.py`. Do not commit customer-identifiable, confidential, or restricted business data.

## Outputs

Running `src.train_pipeline` creates:
- `reports/analytics_summary.json`
- `reports/daily_demand.csv`
- `reports/forecast_results.csv`
- `reports/delivery_risk_metrics.json`
- `reports/delivery_risk_feature_importance.csv`
- `models/delivery_risk_model.joblib`

Generated output files are intentionally not committed by default. Recreate them locally to keep the repository small and reproducible.

## Evaluation approach

- Forecasting: chronological train/test split; MAE and RMSE.
- Delivery risk: stratified train/test split; precision, recall, F1, ROC-AUC when available.
- Explainability: permutation importance on held-out data.
- Deep learning: chronological holdout for one SKU.

Always compare against a simple baseline. For production use, consider time-based cross-validation, demand intermittency, holidays/promotions, stockout-censored demand, data drift, calibration, and operational constraints.

## Suggested GitHub presentation

1. Generate data and run the pipeline.
2. Add screenshots of the dashboard to `reports/screenshots/` after running it.
3. Replace synthetic data with a public dataset and document its source/licence.
4. Record your own evaluated metrics and limitations in the README.
5. Add a short demo GIF or video once the dashboard is working locally.

## Limitations and responsible use

- The default data is synthetic and designed to exercise the code paths.
- This project is a learning/demo framework, not an inventory-management system.
- The late-delivery label is derived from generated delivery duration; results on synthetic data can be unrealistically easy.
- No model metric is hard-coded or presented as an achieved result.
