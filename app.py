import streamlit as st
import pandas as pd
import pickle

# load historical data (hunain)
df = pd.read_csv('data/20100056.csv')
df = df[df['GEO']=='Toronto, Ontario']
df = df[df['North American Industry Classification System (NAICS)']=='Retail trade [44-45]']
df['date'] = pd.to_datetime(df['REF_DATE'], format='%Y-%m')
ts = df[['date','VALUE']].rename(columns={'VALUE':'sales'}).set_index('date')

# load model (hunain)
with open('lgbm_model.pkl','rb') as f:
    model = pickle.load(f)

st.title("Toronto Retail‐Trade Sales Forecast")
st.line_chart(ts, height=300, use_container_width=True)

horizon = st.slider("Months to forecast", 1, 24, 12)
if st.button("Forecast"):
    last = ts.index.max()
    idx  = pd.date_range(last+pd.DateOffset(months=1), periods=horizon, freq='MS')
    future = pd.DataFrame(index=idx)
    future['month']   = idx.month
    future['quarter'] = idx.quarter
    future['year']    = idx.year
    future['lag_1']   = ts.iloc[-1,0]
    future['lag_12']  = ts.iloc[-12,0] if len(ts)>=12 else future['lag_1']
    future['roll_3']  = ts.iloc[-3:,0].mean() if len(ts)>=3 else future['lag_1']
    future['forecast'] = model.predict(future)

    st.line_chart(future[['forecast']], height=300, use_container_width=True)
    st.write("**Top 3 forecast months:**")
    st.write(future['forecast'].nlargest(3))
