# Level 3 — Task 1: Time Series Analysis & Sales Forecasting

## Project Overview
This project aggregates multi-country transaction revenue (`sales_data.csv`, 700 records) into a regular monthly time series spanning **September 2018 – December 2019 (16 periods)**, decomposes the series into trend, seasonal, and residual components, compares smoothing baselines, and evaluates **ARIMA(1,1,1)** against **SARIMA(1,1,1)(1,0,0,4)** on a chronological holdout horizon before generating Q1 2020 revenue forecasts.

## Methodology
1. **Temporal Aggregation & Stationarity:** Parsed `DD/MM/YYYY` dates and resampled total monthly sales (`Freq: MS`). Augmented Dickey-Fuller (ADF) testing yielded `ADF Statistic = -5.1914` (`p = 0.0000`), confirming statistical stationarity.
2. **Decomposition & Smoothing:** Applied additive seasonal decomposition alongside a **3-Month Simple Moving Average (SMA)** and **Holt-Winters Exponential Smoothing**, revealing steady upward trend growth from ~$6.7M to >$7.9M with sharp Q4 promotional peaks.
3. **Chronological Holdout Validation:** Trained models on the first 12 months (`Sep 2018 – Aug 2019`) and evaluated forecasts on the unseen 4-month holdout (`Sep – Dec 2019`).

## Holdout Evaluation Results (Sep – Dec 2019)
| Model | Test RMSE ($) | Test MAE ($) | AIC |
| :--- | :---: | :---: | :---: |
| **ARIMA (1,1,1)** | $4,645,057.37 | $3,497,165.54 | 362.04 |
| **SARIMA (1,1,1)(1,0,0,4)** | **$4,281,064.46** | **$3,427,784.74** | 362.19 |

## Out-of-Sample Forecast (Q1 2020)
* **January 2020:** `$6,428,094.14`
* **February 2020:** `$11,707,602.23`
* **March 2020:** `$7,006,683.41`

## Tech Stack
`Python 3.12` • `Pandas` • `NumPy` • `Statsmodels` • `Scikit-learn` • `Matplotlib`