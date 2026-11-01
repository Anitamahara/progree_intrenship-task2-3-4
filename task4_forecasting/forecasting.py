import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load data (synthetic monthly sales; swap in your own CSV if you have one) ----------
rng = np.random.default_rng(42)
idx = pd.date_range("2015-01-01", periods=120, freq="MS")
t = np.arange(120)
temp = 20 + 10 * np.sin(2 * np.pi * (t - 3) / 12) + rng.normal(0, 1.5, 120)
promo = rng.binomial(1, 0.25, 120)
sales = 200 + 2 * t + 30 * np.sin(2 * np.pi * t / 12) + 4 * temp + 25 * promo + rng.normal(0, 8, 120)
df = pd.DataFrame({"sales": sales, "temp": temp, "promo": promo}, index=idx)
df.to_csv("sales_data.csv")

# 2. Stationarity check
def adf(s, name):
    p = adfuller(s.dropna())[1]
    print(f"{name:25s} p={p:.4f} -> {'stationary' if p < 0.05 else 'NOT stationary'}")

adf(df.sales, "raw")
adf(df.sales.diff(), "diff(1)")
adf(df.sales.diff(12), "seasonal diff(12)")
adf(df.sales.diff().diff(12), "diff(1) + diff(12)")

seasonal_decompose(df.sales, model="additive", period=12).plot()
plt.savefig("decomposition.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. SARIMA forecast
train, test = df.sales[:-12], df.sales[-12:]
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
fc = model.get_forecast(12)
pred = fc.predicted_mean
ci = fc.conf_int()

mae = mean_absolute_error(test, pred)
rmse = np.sqrt(mean_squared_error(test, pred))
print(f"SARIMA  MAE={mae:.2f}  RMSE={rmse:.2f}")

plt.figure(figsize=(9, 4))
plt.plot(train[-36:], label="Train")
plt.plot(test, label="Actual")
plt.plot(pred, label="Forecast")
plt.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.25, label="95% CI")
plt.legend()
plt.title("SARIMA: Test Forecast")
plt.savefig("sarima_test.png", dpi=150, bbox_inches="tight")
plt.close()

# Future forecast (refit on all data)
full = SARIMAX(df.sales, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
ffc = full.get_forecast(12)
fci = ffc.conf_int()
plt.figure(figsize=(9, 4))
plt.plot(df.sales[-36:], label="History")
plt.plot(ffc.predicted_mean, label="Future forecast")
plt.fill_between(fci.index, fci.iloc[:, 0], fci.iloc[:, 1], alpha=0.25, label="95% CI")
plt.legend()
plt.title("SARIMA: Next 12 Months")
plt.savefig("sarima_future.png", dpi=150, bbox_inches="tight")
plt.close()

# 4. Interpretability: Random Forest + SHAP
feat = pd.DataFrame({
    "lag1": df.sales.shift(1),
    "lag12": df.sales.shift(12),
    "month": df.index.month,
    "temp": df.temp,
    "promo": df.promo,
    "trend": t,
}, index=idx).dropna()
y = df.sales.loc[feat.index]

rf = RandomForestRegressor(n_estimators=300, random_state=42).fit(feat[:-12], y[:-12])
print("RF MAE:", round(mean_absolute_error(y[-12:], rf.predict(feat[-12:])), 2))

sv = shap.TreeExplainer(rf).shap_values(feat)
shap.summary_plot(sv, feat, show=False)
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
shap.summary_plot(sv, feat, plot_type="bar", show=False)
plt.savefig("shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()

print("Mean |SHAP| per feature:")
print(pd.Series(np.abs(sv).mean(axis=0), index=feat.columns).sort_values(ascending=False).round(2))