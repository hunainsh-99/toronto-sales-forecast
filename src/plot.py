import pandas as pd
import matplotlib.pyplot as plt

# load & filter (hunain)
df = pd.read_csv('data/20100056.csv')
df = df[df['GEO']=='Toronto, Ontario']
df = df[df['North American Industry Classification System (NAICS)']=='Retail trade [44-45]']

# build time series (hunain)
df['date'] = pd.to_datetime(df['REF_DATE'], format='%Y-%m')
ts = df[['date','VALUE']].rename(columns={'VALUE':'sales'}).set_index('date')

# plot & save (hunain)
plt.figure(figsize=(10,4))
plt.plot(ts.index, ts['sales'])
plt.title('Toronto Monthly Retail-Trade Sales')
plt.xlabel('Date')
plt.ylabel('Sales ($000s)')
plt.tight_layout()
plt.savefig('sales_plot.png')
print("✅ Saved plot → sales_plot.png")
