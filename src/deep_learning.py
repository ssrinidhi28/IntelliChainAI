"""Optional LSTM forecasting experiment (requires PyTorch)."""
from __future__ import annotations
import argparse
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data_utils import load_orders
from src.forecasting import make_daily_series

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/supply_chain_demo.csv")
    parser.add_argument("--target-product", default="SKU-001")
    parser.add_argument("--lookback", type=int, default=28)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit("PyTorch is optional. Install it with `pip install torch` to run this experiment.") from exc

    torch.manual_seed(42)
    np.random.seed(42)
    df = load_orders(args.input)
    product = args.target_product if args.target_product in set(df["product_id"]) else str(df["product_id"].mode().iloc[0])
    series = make_daily_series(df, product)["demand"].to_numpy(dtype=np.float32)
    if len(series) <= args.lookback + 30:
        raise SystemExit("Not enough history for LSTM experiment. Generate more days of data.")
    split = int(len(series) * 0.8)
    train_series = series[:split]
    mean, std = float(train_series.mean()), float(train_series.std() + 1e-6)
    scaled = (series - mean) / std

    def sequences(values, start, stop):
        X, y = [], []
        for i in range(max(start, args.lookback), stop):
            X.append(values[i-args.lookback:i])
            y.append(values[i])
        return np.asarray(X, dtype=np.float32)[..., None], np.asarray(y, dtype=np.float32)
    X_train, y_train = sequences(scaled, args.lookback, split)
    X_test, y_test = sequences(scaled, split, len(scaled))
    if len(X_train) == 0 or len(X_test) == 0:
        raise SystemExit("Unable to create train/test sequences; adjust lookback or dataset length.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    class LSTMForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
            self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    model = LSTMForecaster().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.MSELoss()
    Xt = torch.tensor(X_train, device=device)
    yt = torch.tensor(y_train, device=device)
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(X_test, device=device)).cpu().numpy()
    pred = np.maximum(0, pred_scaled * std + mean)
    actual = np.maximum(0, y_test * std + mean)
    print(f"Product: {product} | device: {device} | epochs: {args.epochs}")
    print(f"Holdout MAE: {mean_absolute_error(actual, pred):.3f}")
    print(f"Holdout RMSE: {math.sqrt(mean_squared_error(actual, pred)):.3f}")
    print("Compare this chronological holdout against the naive and scikit-learn baselines before drawing conclusions.")

if __name__ == "__main__":
    main()
