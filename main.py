import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Step 1: Generate synthetic data
np.random.seed(42)
T = 200
time = np.arange(T)

data = 10 + 0.05*time + 2*np.sin(2*np.pi*time/12) + np.random.normal(0, 0.5, T)

df = pd.DataFrame({'y': data})

# Step 2: SARIMA model
sarima_model = SARIMAX(df['y'], order=(1,1,1), seasonal_order=(1,1,1,12))
sarima_fit = sarima_model.fit(disp=False)
sarima_pred = sarima_fit.predict(start=1, end=len(df)-1)

# Step 3: Feature engineering for ML
df['lag1'] = df['y'].shift(1)
df['lag12'] = df['y'].shift(12)
df = df.dropna()

X = df[['lag1', 'lag12']]
y = df['y']

# Step 4: ML model
ml_model = GradientBoostingRegressor()
ml_model.fit(X, y)
ml_pred = ml_model.predict(X)

# Step 5: Hybrid model
hybrid_pred = (ml_pred + sarima_pred[-len(ml_pred):]) / 2

# Step 6: Plot results
plt.plot(y.values, label="True")
plt.plot(ml_pred, label="ML", linestyle='--')
plt.plot(hybrid_pred, label="Hybrid", linestyle=':')
plt.title("Hybrid Time Series Forecasting")
plt.legend()
plt.show()
