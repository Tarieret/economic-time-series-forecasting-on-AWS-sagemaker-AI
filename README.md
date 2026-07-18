# Economic Time-Series Forecasting on AWS SageMaker

*Inflation forecasts inform rate decisions, budget planning, and investment allocation; a model that's off by even a percentage point on CPI can misdirect millions in portfolio or policy decisions.*

**Problem:** Forecasting U.S. Consumer Price Index (CPI) requires choosing between classical statistical methods and deep learning, each with different failure modes on small, non-stationary monthly economic data. The goal was to identify which approach holds up under rigorous time-series validation and deploy the winner as a production-ready endpoint, not just a notebook result.

**Architecture decision:** Started with LSTM as the presumed strongest candidate, then pivoted to XGBoost after testing showed deep learning struggled with CPI's small monthly sample size and non-stationary trend. Chose a tree-based approach over statistical baselines (ARIMA, Prophet) because it could absorb engineered lag features without requiring the series to be fully stationary by construction. Applied first-order differencing so the model predicts monthly delta instead of absolute CPI level, directly targeting the non-stationarity problem rather than working around it. Deployment was packaged separately from the notebook (`model.tar.gz` plus a custom inference handler for SageMaker) to keep cloud costs isolated from experimentation.

**What broke:** LSTM underperformed on the monthly CPI series, the sample size was too small and the trend too non-stationary for it to generalize, which is what triggered the pivot to XGBoost. Once training on raw CPI levels was replaced with first-order differencing, the model captured actual inflationary momentum rather than fitting a shifting baseline.

**Metric:** XGBoost achieved MAE 1.83 / RMSE 2.0 on a 24-month holdout set (Dec 2023–Nov 2025), against Prophet (MAE 7.09 / RMSE 7.15) and ARIMA(1,1,1) (MAE 8.04 / RMSE 9.29). A roughly 4x error reduction over the next-best baseline.

## Model Comparison Results
![Model Comparison](Images/RMSE_Comparison.png)

## 🏆 Champion Model: XGBoost Performance
![XGBoost CPI Forecast](Images/XGBoost_CPI.png)

## Data
- Source: Federal Reserve Economic Data (FRED)
- Series: CPIAUCSL
- Frequency: Monthly
- Evaluation window: 24-month holdout (Dec 2023 – Nov 2025)

## Methods
- ARIMA (baseline statistical model)
- Prophet (trend + seasonality model)
- XGBoost (feature-engineered regressor with stationarity transformation)

## Results

| Model             | MAE   | RMSE  |
|------------------|-------|-------|
| XGBoost (Winner)  | 1.83  | 2.0   |
| Prophet           | 7.09  | 7.15  |
| ARIMA (1,1,1)     | 8.04  | 9.29  |

## Technical Approach
- First-order differencing to remove trend bias; model predicts monthly delta, not absolute CPI level
- 12-month sliding window of lag features to capture seasonal momentum and annual cycles
- ARIMA and Prophet benchmarked as statistical baselines before training XGBoost
- Final model serialized to `model.tar.gz` with a custom inference handler for SageMaker deployment
- Deployment script kept separate from the notebook to avoid unnecessary cloud costs

## Reflection
Model selection should follow the data's constraints, not the other way around. A simpler, well-regularized architecture outperformed a more complex one once the dataset's scale and non-stationarity were properly accounted for.
