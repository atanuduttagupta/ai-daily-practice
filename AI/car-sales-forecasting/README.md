# PA Day 10 — Track J (Real-world problems): Sales forecasting

**Libraries:** `statsforecast` (Nixtla's fast, modern time-series models), pandas, matplotlib. Free and open source, no API key.
**Data:** monthly car sales in Quebec, Jan 1960 to Dec 1968 (108 points), an open dataset downloaded once from GitHub and cached as `monthly-car-sales.csv`.

## What this project does
Forecasts the next 12 months of sales with four models, and backtests them so you know whether to trust the forecast.

- **Input:** none to provide (the CSV downloads automatically; internet needed the first time).
- **Output:** a backtest score table, a 12-month forecast table, and `forecast.png`.

## Run
```bash
pip install -r requirements.txt
python main.py
```

## Theory
**Time series:** values recorded over time. Their order matters, so you can't shuffle rows like in ordinary ML. A series is usually described as:
- **Trend:** long-term rise or fall.
- **Seasonality:** a pattern that repeats at a fixed period (here, every 12 months).
- **Noise:** everything unexplained.

**Forecasting** means predicting future values for a **horizon** (here 12 months). Businesses use it for inventory, staffing and budgets.

**Always start with a baseline.** *Naive* repeats the last value; *seasonal naive* repeats the value from the same period last year. If a fancy model can't beat these, it isn't adding value.

**Models used here** (the "Auto" ones search for their own settings):
- **ETS** (exponential smoothing): weights recent data more heavily and models level, trend and seasonality.
- **ARIMA:** models a value from its own past values and past errors.
- **Theta:** a simple, robust method that is often surprisingly strong.

**Backtesting (time-series cross-validation):** pretend you are at an earlier date, forecast the next 12 months, compare with what really happened, then repeat at other dates. Rule: **never train on data from after the period you predict** (that is leakage), and never shuffle. Here: 3 windows of 12 months.

**Metrics:** *MAE* (average error, in sales units, easy to read), *RMSE* (punishes big misses more), *MAPE* (percentage error; unstable when values are near zero). Pick one that matches the business cost of being wrong.

**Beyond this day:** forecasts should come with **uncertainty** (prediction intervals; statsforecast supports `level=[80, 95]`). With many series or extra signals (promotions, holidays, price), ML models such as LightGBM, or newer deep-learning and pretrained time-series models (for example Chronos or TimesFM), can help.

## What the output shows (one run)
| Model | MAE | RMSE | MAPE % |
|---|---|---|---|
| SeasonalNaive (baseline) | 1,462 | 1,811 | 8.4 |
| AutoARIMA | 1,639 | 1,986 | 10.2 |
| AutoETS | 1,890 | 2,444 | 11.6 |
| AutoTheta | 2,456 | 2,992 | 14.5 |

- **The simple "same as last year" baseline won**, beating all three automatic models. This is exactly why you backtest against a baseline before trusting a fancier model.
- The forecasts agree on the shape (low in winter, peak in May, a bump in October) but disagree on levels, mostly because the series trends upward and the models treat that trend differently.

**Be careful:** this is one series, 108 points, and only 3 backtest windows, so the ranking could change with other dates, and results may differ with newer library versions. It shows a method, not a general truth about the models. The data is also decades old.

## Try these
1. Increase `n_windows` to 5 and see whether the ranking changes.
2. Add `level=[80, 95]` to `sf.forecast(...)` and plot the prediction intervals.
3. Add more models (`AutoCES`, `HoltWinters`, `Naive`) and compare.
4. Replace the CSV with your own sales data (columns `Month`, `Sales`).
5. Forecast with LightGBM using lag features (the `mlforecast` library) and compare.

## Likely interview questions
- How do you validate a time-series model, and why can't you use random k-fold?
- What baseline would you compare against? What is seasonality?
- MAE vs RMSE vs MAPE: when would you use each?

## One thing learned
A forecast is only as trustworthy as its backtest: here the simplest baseline beat the "smart" models, which you only discover by testing it.
