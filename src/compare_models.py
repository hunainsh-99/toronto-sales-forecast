import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import lightgbm as lgb

# 1. load engineered features (hunain)
df = pd.read_csv('engineered_features.csv', index_col='date', parse_dates=True)
X = df.drop('sales', axis=1)
y = df['sales']

# 2. split train/test (hunain)
split_date = df.index.max() - pd.DateOffset(months=12)
X_train = X.loc[:split_date]
y_train = y.loc[:split_date]
X_test = X.loc[split_date + pd.DateOffset(months=1):]
y_test = y.loc[split_date + pd.DateOffset(months=1):]

def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{name}\tRMSE: {rmse:.2f}\tMAE: {mae:.2f}")

# 3. ARIMA benchmark (hunain)
print("Evaluating ARIMA model...")
arima = ARIMA(y_train, order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
arima_pred = arima.forecast(steps=len(y_test))
evaluate_model('ARIMA', y_test, arima_pred)

# 4. Prophet benchmark (hunain)
print("\nEvaluating Prophet model...")
df_prop = y_train.reset_index().rename(columns={'date':'ds','sales':'y'})
prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
prophet.fit(df_prop)
future = prophet.make_future_dataframe(periods=len(y_test), freq='MS')
fcst = prophet.predict(future).set_index('ds')[['yhat']]
prophet_pred = fcst['yhat'][-len(y_test):]
evaluate_model('Prophet', y_test, prophet_pred)

# 5. LightGBM baseline (hunain)
print("\nEvaluating LightGBM model...")
with open('lgbm_model.pkl', 'rb') as f:
    lgbm_model = pickle.load(f)

lgbm_pred = lgbm_model.predict(X_test)
evaluate_model('LightGBM', y_test, lgbm_pred)
