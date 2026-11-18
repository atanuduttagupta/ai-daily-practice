"""
PA - Day 10 - Track J (Real-world problems): Subtopic 1 - Sales forecasting
Libraries: statsforecast (Nixtla: fast, modern time-series models) + pandas + matplotlib.
Data: monthly car sales in Quebec, 1960-1968 (open dataset, downloaded once and cached).

Goal: forecast the next 12 months of sales, and prove the forecast is worth trusting by
backtesting against a simple "same as last year" baseline. No API key, no paid service.

Setup:  pip install -r requirements.txt
Run:    python main.py        (needs internet the first time to download the CSV)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, SeasonalNaive

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv"
CACHE = "monthly-car-sales.csv"
H = 12  # forecast horizon: 12 months

# 1) Load data into the format statsforecast expects: unique_id, ds (date), y (value)
if not os.path.exists(CACHE):
    pd.read_csv(URL).to_csv(CACHE, index=False)
raw = pd.read_csv(CACHE)
df = pd.DataFrame({"unique_id": "car_sales", "ds": pd.to_datetime(raw["Month"]), "y": raw["Sales"]})
print(f"Loaded {len(df)} monthly points: {df.ds.min():%Y-%m} to {df.ds.max():%Y-%m}")
print(f"Average monthly sales: {df.y.mean():,.0f}   min {df.y.min():,}   max {df.y.max():,}\n")

# 2) Models. season_length=12 tells them the pattern repeats every 12 months.
models = [
    SeasonalNaive(season_length=12),   # baseline: "next June = last June"
    AutoETS(season_length=12),         # exponential smoothing, picks its own form
    AutoARIMA(season_length=12),       # ARIMA, picks its own orders
    AutoTheta(season_length=12),       # Theta method: strong, simple, robust
]
sf = StatsForecast(models=models, freq="MS", n_jobs=1)

# 3) Backtest: pretend we are at 3 past dates, forecast 12 months ahead, compare to what happened
cv = sf.cross_validation(df=df, h=H, n_windows=3, step_size=H)
names = [m.alias for m in models]
rows = []
for name in names:
    err = cv["y"] - cv[name]
    err = err.astype(float)
    rows.append({"model": name, "MAE": err.abs().mean(), "RMSE": np.sqrt((err ** 2).mean()),
                 "MAPE %": (err.abs() / cv["y"]).mean() * 100})
scores = pd.DataFrame(rows).sort_values("MAE").round({"MAE": 0, "RMSE": 0, "MAPE %": 1})
print(f"=== Backtest: 3 windows x {H} months ===")
print(scores.to_string(index=False))
best = scores.iloc[0]["model"]
base = scores.set_index("model").loc["SeasonalNaive", "MAE"]
print(f"\nBest by MAE: {best}. Baseline (SeasonalNaive) MAE: {base:,.0f}\n")

# 4) Final forecast: refit on all data, predict the next 12 months
fc = sf.forecast(df=df, h=H)
print(f"=== Forecast for the next {H} months (sales) ===")
print(fc.drop(columns="unique_id").round(0).to_string(index=False))

# 5) Plot
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(df.ds, df.y, color="black", label="history")
for name in names:
    ax.plot(fc.ds, fc[name], label=name)
ax.set_title("Monthly car sales: history and 12-month forecasts"); ax.legend(ncol=3)
plt.tight_layout(); plt.savefig("forecast.png", dpi=110)
print("\nSaved forecast.png")
