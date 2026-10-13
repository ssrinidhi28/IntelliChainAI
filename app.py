"""Streamlit dashboard for IntelliChain AI."""
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from src.generate_data import generate_data
from src.data_utils import clean_orders, analytics_summary
from src.forecasting import make_daily_series, evaluate_forecast
from src.risk_model import train_delivery_risk

st.set_page_config(page_title="IntelliChain AI", page_icon="📦", layout="wide")
st.title("📦 IntelliChain AI")
st.caption("Supply-chain analytics • demand forecasting • delivery-risk intelligence")
DATA_PATH = Path("data/supply_chain_demo.csv")

@st.cache_data
def get_data():
    if DATA_PATH.exists():
        return clean_orders(pd.read_csv(DATA_PATH, parse_dates=["order_date", "ship_date", "delivery_date"]))
    return clean_orders(generate_data(days=540, seed=42))

df = get_data()
with st.sidebar:
    st.header("Filters")
    date_min = df["order_date"].min().date()
    date_max = df["order_date"].max().date()
    date_range = st.date_input("Order date range", value=(date_min, date_max), min_value=date_min, max_value=date_max)
    categories = sorted(df["category"].dropna().unique().tolist())
    selected_categories = st.multiselect("Category", categories, default=categories)
    regions = sorted(df["region"].dropna().unique().tolist())
    selected_regions = st.multiselect("Region", regions, default=regions)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    view = df[df["order_date"].dt.date.between(start_date, end_date)]
else:
    view = df.copy()
if selected_categories:
    view = view[view["category"].isin(selected_categories)]
if selected_regions:
    view = view[view["region"].isin(selected_regions)]

if view.empty:
    st.warning("No rows match the current filters. Change the filters in the sidebar.")
    st.stop()

summary = analytics_summary(view)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Orders", f"{summary['orders']:,}")
c2.metric("Revenue", f"${summary['revenue']:,.0f}")
c3.metric("Estimated margin", f"${summary['estimated_margin']:,.0f}")
c4.metric("Late delivery rate", f"{summary['late_delivery_rate']:.1%}")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Demand forecast", "Delivery risk", "Data explorer"])
with tab1:
    left, right = st.columns(2)
    daily = view.groupby("order_date", as_index=False).agg(revenue=("revenue", "sum"), units=("quantity", "sum"))
    left.plotly_chart(px.line(daily, x="order_date", y="revenue", title="Daily revenue"), use_container_width=True)
    category_perf = view.groupby("category", as_index=False).agg(revenue=("revenue", "sum"), units=("quantity", "sum"))
    right.plotly_chart(px.bar(category_perf, x="category", y="revenue", title="Revenue by category"), use_container_width=True)
    region_perf = view.groupby("region", as_index=False).agg(late_rate=("is_late", "mean"), orders=("order_id", "nunique"))
    st.plotly_chart(px.bar(region_perf, x="region", y="late_rate", title="Late delivery rate by region",
                           hover_data=["orders"]), use_container_width=True)
    st.caption("Revenue and margin use the dataset's unit price/cost estimates. Synthetic-data insights are illustrative.")

with tab2:
    products = sorted(view["product_id"].unique())
    product = st.selectbox("Product / SKU", products, index=0)
    horizon = st.slider("Holdout horizon (days)", 7, 56, 28, 7)
    try:
        series = make_daily_series(view, product)
        preds, metrics = evaluate_forecast(series, horizon=horizon)
        st.plotly_chart(px.line(preds, x="date", y=["actual", "predicted"], color="model",
                                title=f"Chronological holdout: {product}"), use_container_width=True)
        st.dataframe(pd.DataFrame(metrics["metrics"]).sort_values("mae"), use_container_width=True)
        st.caption("This chart evaluates held-out historical dates; it is not a future forecast. Train/test periods are chronological.")
        st.download_button("Download forecast evaluation CSV", preds.to_csv(index=False), "forecast_evaluation.csv", "text/csv")
    except ValueError as exc:
        st.info(str(exc))

with tab3:
    st.write("Train an illustrative model on the filtered rows. For real use, validate temporal leakage and label quality.")
    if st.button("Train delivery-risk model", type="primary"):
        try:
            with st.spinner("Training model and calculating held-out metrics..."):
                model, metrics, importance = train_delivery_risk(view)
            st.session_state["risk_metrics"] = metrics
            st.session_state["risk_importance"] = importance
        except Exception as exc:
            st.error(f"Could not train the model: {exc}")
    if "risk_metrics" in st.session_state:
        metrics = st.session_state["risk_metrics"]
        cols = st.columns(5)
        for col, key in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
            val = metrics.get(key)
            col.metric(key.replace("_", " ").title(), "N/A" if val is None else f"{val:.3f}")
        importance = st.session_state["risk_importance"]
        st.plotly_chart(px.bar(importance.head(12).sort_values("importance_mean"),
                               x="importance_mean", y="feature", orientation="h",
                               title="Permutation feature importance (F1 scoring)"), use_container_width=True)
        st.caption(f"Held-out rows: {metrics['test_rows']}. Metrics describe this dataset and split only.")
    else:
        st.info("Click the button to fit and evaluate the model.")

with tab4:
    st.dataframe(view.head(500), use_container_width=True)
    st.download_button("Download filtered data CSV", view.to_csv(index=False), "filtered_orders.csv", "text/csv")
    st.write("Data quality snapshot")
    quality = pd.DataFrame({"column": view.columns, "missing_values": [int(view[c].isna().sum()) for c in view.columns],
                            "missing_pct": [float(view[c].isna().mean() * 100) for c in view.columns]})
    st.dataframe(quality, use_container_width=True)
