import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import pickle

# 1. load engineered features (hunain)
fe = pd.read_csv('engineered_features.csv', index_col='date', parse_dates=True)

# 2. define X and y
X = fe.drop('sales', axis=1)
y = fe['sales']

# 3. time-based split: keep last 12 months as test (hunain)
split_date = fe.index.max() - pd.DateOffset(months=12)
X_train = X.loc[:split_date]
X_test  = X.loc[split_date + pd.DateOffset(days=1):]
y_train = y.loc[:split_date]
y_test  = y.loc[split_date + pd.DateOffset(days=1):]

# 4. train LightGBM (hunain)
dtrain = lgb.Dataset(X_train, label=y_train)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1
}
model = lgb.train(params, dtrain, num_boost_round=200)

# 5. predict & evaluate (hunain)
import numpy as np
y_pred = model.predict(X_test)
mse   = mean_squared_error(y_test, y_pred)
rmse  = np.sqrt(mse)
mae   = mean_absolute_error(y_test, y_pred)
print(f'LightGBM → RMSE: {rmse:.2f}, MAE: {mae:.2f}')


print(f'LightGBM → RMSE: {rmse:.2f}, MAE: {mae:.2f}')

# 6. save the model (hunain)
with open('lgbm_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print('✅ Model saved to lgbm_model.pkl')
