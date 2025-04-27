import pandas as pd
import pickle

# load your engineered features (hunain)
fe = pd.read_csv('engineered_features.csv', index_col='date', parse_dates=True)

# load your trained model (hunain)
with open('lgbm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# make features (hunain)
last_date = fe.index.max()
future_idx = pd.date_range(
    last_date + pd.DateOffset(months=1),
    last_date + pd.DateOffset(months=12),
    freq='MS'
)
future = pd.DataFrame(index=future_idx)
future['month']   = future.index.month
future['quarter'] = future.index.quarter
future['year']    = future.index.year
future['lag_1']   = fe['sales'].iloc[-1]
future['lag_12']  = fe['sales'].iloc[-12]
future['roll_3']  = fe['sales'].iloc[-3:].mean()

# predict (hunain)
future['forecast'] = model.predict(future)

# show top 3 peaks (hunain)
peaks = future['forecast'].nlargest(3)
print("Top 3 forecast months (sales):")
print(peaks)
