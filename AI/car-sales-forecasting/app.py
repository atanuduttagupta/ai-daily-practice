import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, SeasonalNaive

warnings.filterwarnings("ignore")

DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv"
CACHE_FILE = Path(__file__).with_name("monthly-car-sales.csv")

st.set_page_config(page_title="Monthly Car Sales Forecasting Lab", page_icon="🚗", layout="wide")
st.title("🚗 Monthly Car Sales Forecasting Lab")
st.caption("PA Day 10 · Track J — Backtesting / Time-Series Cross-Validation")
st.write(
    "Forecast monthly car sales, then test whether the models would have made "
    "useful predictions at earlier points in time. The data is never shuffled."
)

@st.cache_data(show_spinner=False)
def load_data():
    if CACHE_FILE.exists():
        raw = pd.read_csv(CACHE_FILE)
    else:
        raw = pd.read_csv(DATA_URL)
        raw.to_csv(CACHE_FILE, index=False)
    if not {"Month", "Sales"}.issubset(raw.columns):
        raise ValueError("CSV must contain Month and Sales columns.")
    data = pd.DataFrame({
        "unique_id": "car_sales",
        "ds": pd.to_datetime(raw["Month"]),
        "y": pd.to_numeric(raw["Sales"], errors="coerce"),
    }).dropna().sort_values("ds").reset_index(drop=True)
    return data

try:
    with st.spinner("Loading monthly car sales data..."):
        df = load_data()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.info("The CSV downloads on first run. Check internet access or place monthly-car-sales.csv beside app.py.")
    st.stop()

st.sidebar.header("Experiment settings")
horizon = st.sidebar.slider("Forecast horizon (months)", 1, 24, 12)
season_length = st.sidebar.selectbox("Season length", [12, 6, 4, 3, 1], index=0)
step_size = st.sidebar.slider("Step size (months)", 1, 24, horizon)
max_windows = max(1, min(8, (len(df) - max(24, horizon)) // step_size + 1))
n_windows = st.sidebar.slider("Backtest windows", 1, max_windows, min(3, max_windows))

st.sidebar.markdown("---")
st.sidebar.subheader("Models")
selected = []
if st.sidebar.checkbox("Seasonal Naive (baseline)", True):
    selected.append(SeasonalNaive(season_length=season_length))
if st.sidebar.checkbox("AutoARIMA", True):
    selected.append(AutoARIMA(season_length=season_length))
if st.sidebar.checkbox("AutoETS", True):
    selected.append(AutoETS(season_length=season_length))
if st.sidebar.checkbox("AutoTheta", True):
    selected.append(AutoTheta(season_length=season_length))
if not selected:
    st.warning("Select at least one model in the sidebar.")
    st.stop()

model_names = [m.alias for m in selected]
min_train = len(df) - (n_windows - 1) * step_size - horizon
if min_train < max(24, season_length * 2):
    st.error("These settings leave too little training data. Reduce the windows, step size, or horizon.")
    st.stop()

# Data overview
st.subheader("1. Historical data")
a, b, c, d = st.columns(4)
a.metric("Observations", f"{len(df):,}")
b.metric("From", df.ds.min().strftime("%b %Y"))
c.metric("To", df.ds.max().strftime("%b %Y"))
d.metric("Average monthly sales", f"{df.y.mean():,.0f}")

with st.expander("View data table"):
    st.dataframe(df[["ds", "y"]].rename(columns={"ds": "Month", "y": "Sales"}), use_container_width=True)

fig, ax = plt.subplots(figsize=(12, 3.5))
ax.plot(df.ds, df.y, marker="o", markersize=3)
ax.set(title="Monthly car sales — historical observations", xlabel="Month", ylabel="Sales")
ax.grid(alpha=.25)
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.info("Time-series rule: preserve chronological order. Randomly shuffling months can cause future-data leakage.")

# Backtesting
st.subheader("2. Backtesting — test forecasts on the past")
st.write(
    f"Each window predicts {horizon} month(s). The forecast origin advances by "
    f"{step_size} month(s), for {n_windows} historical tests."
)
if st.button("▶ Run backtesting", type="primary") or "cv_result" not in st.session_state:
    try:
        with st.spinner("Running time-series cross-validation..."):
            sf = StatsForecast(models=selected, freq="MS", n_jobs=1)
            cv = sf.cross_validation(
                df=df, h=int(horizon), n_windows=int(n_windows), step_size=int(step_size)
            )
            rows = []
            for name in model_names:
                err = (cv["y"] - cv[name]).astype(float)
                actual = cv["y"].astype(float)
                nonzero = actual.abs() > 1e-12
                rows.append({
                    "Model": name,
                    "MAE": err.abs().mean(),
                    "RMSE": np.sqrt((err ** 2).mean()),
                    "MAPE (%)": (err[nonzero].abs() / actual[nonzero].abs()).mean() * 100 if nonzero.any() else np.nan,
                })
            scores = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
            st.session_state["cv_result"] = cv
            st.session_state["score_result"] = scores
            st.session_state["cv_config"] = (horizon, n_windows, step_size, season_length, tuple(model_names))
    except Exception as exc:
        st.error(f"Backtesting failed: {exc}")
        st.stop()

config = (horizon, n_windows, step_size, season_length, tuple(model_names))
if st.session_state.get("cv_config") != config:
    st.warning("You changed an experiment setting. Click Run backtesting to refresh the results.")
else:
    scores = st.session_state["score_result"]
    st.markdown("#### Backtest scorecard — lower is better")
    formatted = scores.copy()
    formatted["MAE"] = formatted["MAE"].map(lambda x: f"{x:,.1f}")
    formatted["RMSE"] = formatted["RMSE"].map(lambda x: f"{x:,.1f}")
    formatted["MAPE (%)"] = formatted["MAPE (%)"].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}%")
    st.dataframe(formatted, use_container_width=True, hide_index=True)

    winner = scores.iloc[0]
    st.success(f"Best by MAE in these backtest windows: {winner['Model']} — {winner['MAE']:,.1f} sales.")
    if "SeasonalNaive" in scores.Model.values and winner["Model"] == "SeasonalNaive":
        st.info("The simple 'same month last year' baseline won this run. A more complex model is not automatically better.")

    metric = st.selectbox("Metric to visualize", ["MAE", "RMSE", "MAPE (%)"])
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ordered = scores.sort_values(metric)
    ax.bar(ordered.Model, ordered[metric])
    ax.set(title=f"Model comparison — {metric}", ylabel=metric)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("Inspect individual backtest predictions"):
        cv = st.session_state["cv_result"]
        cols = ["ds", "cutoff", "y"] + [n for n in model_names if n in cv.columns]
        st.dataframe(cv[cols].sort_values(["cutoff", "ds"]), use_container_width=True, hide_index=True)

# Final forecast
st.subheader("3. Forecast future monthly sales")
st.write("Once the backtest is reviewed, refit the models on all available observations.")
if st.button("🔮 Generate final forecast"):
    try:
        with st.spinner("Fitting models and forecasting..."):
            sf_final = StatsForecast(models=selected, freq="MS", n_jobs=1)
            forecast = sf_final.forecast(df=df, h=int(horizon))
            st.session_state["forecast_result"] = forecast
            st.session_state["forecast_config"] = config
    except Exception as exc:
        st.error(f"Forecast failed: {exc}")

forecast = st.session_state.get("forecast_result")
if forecast is not None:
    if st.session_state.get("forecast_config") != config:
        st.warning("Settings have changed. Generate the final forecast again.")
    else:
        table = forecast.drop(columns=["unique_id"], errors="ignore").copy()
        table = table.rename(columns={"ds": "Month"})
        for col in table.select_dtypes(include="number").columns:
            table[col] = table[col].round(0)
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button("⬇ Download forecast CSV", table.to_csv(index=False).encode(), "monthly_car_sales_forecast.csv", "text/csv")

        fig, ax = plt.subplots(figsize=(12, 4.2))
        ax.plot(df.ds, df.y, color="black", linewidth=2, label="Historical sales")
        for name in model_names:
            if name in forecast.columns:
                ax.plot(forecast.ds, forecast[name], marker="o", markersize=3, label=name)
        ax.axvline(df.ds.max(), color="gray", linestyle="--", label="Forecast starts")
        ax.set(title=f"Monthly car sales — next {horizon} months", xlabel="Month", ylabel="Sales")
        ax.grid(alpha=.25)
        ax.legend(ncol=2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# Learning tabs
st.subheader("4. Learn and experiment")
t1, t2, t3, t4 = st.tabs(["Backtesting", "Models", "Metrics", "Try next"])
with t1:
    st.markdown("""
    **Backtesting** means pretending we are at an earlier date, forecasting the next
    months, and comparing predictions with what actually happened.

    - **Horizon (`h`):** how far ahead we predict.
    - **Windows (`n_windows`):** how many historical forecast tests we run.
    - **Step size:** how far the forecast origin moves between tests.
    - **Leakage:** using information from after the date we are pretending to forecast from.
    """)
with t2:
    st.markdown("""
    - **Seasonal Naive:** repeats the value from the same month in the previous year.
    - **AutoARIMA:** learns patterns from past values and past forecast errors.
    - **AutoETS:** models level, trend and seasonality, weighting recent observations.
    - **AutoTheta:** a relatively simple forecasting method that can be a strong benchmark.
    """)
with t3:
    st.markdown("""
    - **MAE:** average size of the error, in sales units. Easy to explain.
    - **RMSE:** like MAE, but penalizes large misses more heavily.
    - **MAPE:** average error as a percentage of actual sales; unstable near zero.

    Lower is better for these metrics. Choose the metric that reflects the real cost of being wrong.
    """)
with t4:
    st.markdown("""
    1. Increase the number of backtest windows and see whether the ranking changes.
    2. Try a 6-month rather than 12-month horizon.
    3. Add prediction intervals (for example, 80% and 95%) to express uncertainty.
    4. Replace the sample CSV with your own monthly sales data.
    5. Try LightGBM with lag features and compare it against these statistical models.
    """)

st.divider()
st.caption(
    "This is one historical series with 108 observations and a limited number of backtest "
    "windows. Results depend on the dataset, dates and library versions; they are not a "
    "universal ranking of forecasting algorithms."
)
