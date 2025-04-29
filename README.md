# Toronto Retail-Trade Sales Forecast

An end-to-end forecasting pipeline for Toronto’s monthly retail-trade sales, built with:

- Public **Statistics Canada** data  
- **Feature engineering** (date parts, lags, rolling means)  
- **LightGBM** modeling with ARIMA & Prophet benchmarks  
- An **interactive Streamlit** dashboard

---

##  Project Structure

```text
toronto-sales-forecast/
│
├── data/
│   ├── 20100056.csv            # Raw StatsCan data
│   └── 20100056_MetaData.csv   # Metadata
│
├── engineered_features.csv     # Output of feature_engineering.py
├── lgbm_model.pkl              # Trained LightGBM model
├── sales_plot.png              # Exploratory time-series plot
│
├── src/
│   ├── data_loader.py          # Load & filter raw CSV
│   ├── feature_engineering.py  # Build lags, rolling means, date parts
│   ├── modeling.py             # Train & evaluate LightGBM
│   ├── future_forecast.py      # Produce 12-month forecast
│   ├── compare_models.py       # Benchmark ARIMA, Prophet, LightGBM
│   └── plot.py                 # Generate sales_plot.png
│
├── app.py                      # Streamlit dashboard
├── README.md                   # ← you are here
└── .gitignore


##  Overview

1. **Data Ingestion**  
   * Downloaded StatsCan table **20100056**  
   * Filtered to **Toronto, Ontario [CMA]** & **Retail trade [44-45]**

2. **Feature Engineering**  
   * Date parts: month, quarter, year  
   * Lag features: `lag_1`, `lag_12`  
   * 3-month rolling mean  

3. **Modeling & Evaluation**  
   * Trained LightGBM on engineered features  
   * Hold-out test: latest 12 months  
   * Benchmarks: ARIMA & Prophet  
   * **LightGBM RMSE ≈ 531 k**, **MAE ≈ 402 k**

4. **Forecasting & Insights**  
   * 12-month ahead forecast  
   * Identified peak months for inventory & staffing

5. **Interactive Dashboard**  
   * Streamlit app with horizon slider  
   * Historical + forecast plots

---

##  Getting Started

### Prerequisites
* Python 3.10+  

## Installation
```bash
# clone
git clone https://github.com/<your-user>/toronto-sales-forecast.git
cd toronto-sales-forecast

# env
python3 -m venv .venv
source .venv/bin/activate

# deps
pip install -r requirements.txt

Typical Workflow:

python src/data_loader.py         # clean data
python src/feature_engineering.py # build features
python src/modeling.py            # train LightGBM
python src/compare_models.py      # ARIMA vs Prophet vs LightGBM
python src/future_forecast.py     # 12-month forecast
streamlit run app.py              # open dashboard

The Results:
Model | RMSE (↓) | MAE (↓)
ARIMA | 831 594 | 704 622
Prophet | 1 057 091 | 804 211
LightGBM | 531 195 | 401 793
LightGBM’s extra covariates (lags, rolling stats, seasonality) cut error by ~36 % versus ARIMA.

##  Business Insights
Peak demand in Oct–Dec → stock up & extend staffing.

Early-year lull → run promotions Jan-Mar to smooth revenue.

Use rolling forecast updates monthly to adjust inventory in real time.

streamlit link: https://toronto-sales-forecast-ncpyboqdmamuodwiuo8tyf.streamlit.app/

SHAP explainability in dashboard

Docker image + GitHub Actions CI

Extend to other NAICS categories
