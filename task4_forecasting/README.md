# Task 4: Time-Series Forecasting & Interpretability

**Tools:** Python, Statsmodels, Scikit-Learn, SHAP

## Method
1. ADF test: raw series non-stationary; stationary after diff(1) + seasonal diff(12) (p=0.0001)
2. Seasonal decomposition (decomposition.png)
3. SARIMA(1,1,1)(1,1,1,12) forecast with 95% confidence intervals
4. Random Forest + SHAP to explain which variables drive sales

## Results
- SARIMA: MAE 15.22, RMSE 19.04
- Random Forest: MAE 20.78
- Top SHAP feature: lag12 (same month last year), then trend, lag1, promo, temp

## Run
pip install pandas numpy statsmodels scikit-learn shap matplotlib
python forecasting.py

## Output
decomposition.png, sarima_test.png, sarima_future.png, shap_summary.png, shap_importance.png