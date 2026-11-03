# CPI Forecasting on AWS SageMaker

Inflation forecasts feed into rate decisions, budgets and investment planning. This project compares statistical and machine-learning approaches for forecasting the U.S. Consumer Price Index (CPI) and packages the best model for deployment on AWS SageMaker.

## Key results
24-month holdout (Dec 2023 – Nov 2025):

| Model | MAE | RMSE |
|-------|-----|------|
| **XGBoost** | **1.83** | **2.00** |
| Prophet | 7.09 | 7.15 |
| ARIMA(1,1,1) | 8.04 | 9.29 |

XGBoost had about 4x lower error than the next-best baseline. Errors are in CPI index points.

![Model comparison](Images/RMSE_Comparison.png)
![XGBoost CPI forecast](Images/XGBoost_CPI.png)

## Approach
- **Data:** FRED series CPIAUCSL (monthly).
- **Baselines first:** I fit ARIMA and Prophet as statistical benchmarks before training anything more complex.
- **Differencing:** the model predicts the month-over-month change in CPI, not the CPI level. This handles the upward trend directly instead of fitting a shifting baseline.
- **Lag features:** the previous 12 months of changes are inputs to XGBoost.
- **Deployment:** the final model is saved as `model.tar.gz` with a custom inference handler for SageMaker. The deployment script is separate from the notebook, so experimenting doesn't run up cloud costs.

## What broke
I started with an LSTM, expecting deep learning to perform best. It didn't generalize: about 20 years of monthly data is too small a sample, and the trend too non-stationary. That pushed me to a simpler, feature-engineered XGBoost model, which outperformed both baselines once the series was differenced.

## Stack
Python · pandas · XGBoost · statsmodels (ARIMA) · Prophet · AWS SageMaker · FRED
