import pandas as pd

# 1. load & clean series (hunain)
df = pd.read_csv('data/20100056.csv')
df = df[df['GEO'] == 'Toronto, Ontario']
df = df[df['North American Industry Classification System (NAICS)'] == 'Retail trade [44-45]']
df['date'] = pd.to_datetime(df['REF_DATE'], format='%Y-%m')
ts = df[['date','VALUE']].rename(columns={'VALUE':'sales'}).set_index('date')

# 2. create features (hunain)
fe = ts.copy()
fe['month']     = fe.index.month
fe['quarter']   = fe.index.quarter
fe['year']      = fe.index.year
fe['lag_1']     = fe['sales'].shift(1)
fe['lag_12']    = fe['sales'].shift(12)
fe['roll_3']    = fe['sales'].rolling(window=3).mean()

# 3. drop NaNs (hunain)
fe = fe.dropna()

# 4. save to CSV (hunain)
fe.to_csv('engineered_features.csv')

print("✅ features engineered; saved to engineered_features.csv")
print(fe.head())
