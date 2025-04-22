import streamlit as st
from prophet import Prophet
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Set up the web page configuration
st.set_page_config(page_title="Gold Price Prediction App", layout="centered")
st.title("Gold Price Prediction (Including Global Stock Market Impact)")
st.markdown("This AI model uses Facebook Prophet to predict gold prices, considering the impact of global stock market trends.")

# User selects the forecast period (in days)
period = st.selectbox("Choose forecast period", [7, 14, 30])

# Get data for gold and global stock markets
@st.cache_data
def get_all_data():
    # Gold ETF (GLD)
    gold = yf.download("GLD", start="2015-01-01")[['Close']].rename(columns={"Close": "gold"})
    # S&P500
    sp500 = yf.download("^GSPC", start="2015-01-01")[['Close']].rename(columns={"Close": "sp500"})
    # Nikkei 225
    nikkei = yf.download("^N225", start="2015-01-01")[['Close']].rename(columns={"Close": "nikkei"})
    # DAX
    dax = yf.download("^GDAXI", start="2015-01-01")[['Close']].rename(columns={"Close": "dax"})

    # Merge data
    df = gold.join([sp500, nikkei, dax], how='inner')
    df = df.reset_index().rename(columns={"Date": "ds"})
    df = df.rename(columns={"gold": "y"})  # Prophet requires the target column to be "y"
    return df

df = get_all_data()

# Show historical gold price vs stock market trends
st.subheader("Historical Gold Prices vs Global Stock Markets")
st.line_chart(df.set_index("ds")[["y", "sp500", "nikkei", "dax"]])

# Create Prophet model with external regressors (stock market data)
model = Prophet(daily_seasonality=True)
model.add_regressor("sp500")
model.add_regressor("nikkei")
model.add_regressor("dax")

# Fit the model
model.fit(df)

# Create future dataframe
future = model.make_future_dataframe(periods=period)

# Add future stock market data (using the most recent data)
latest_sp500 = df["sp500"].iloc[-1]
latest_nikkei = df["nikkei"].iloc[-1]
latest_dax = df["dax"].iloc[-1]

future["sp500"] = df["sp500"].tolist() + [latest_sp500]*period
future["nikkei"] = df["nikkei"].tolist() + [latest_nikkei]*period
future["dax"] = df["dax"].tolist() + [latest_dax]*period

# Make prediction
forecast = model.predict(future)

# Show prediction plot
st.subheader("Prediction Chart")
fig1 = model.plot(forecast)
st.pyplot(fig1)

# Show prediction table
forecast_tail = forecast[['ds', 'yhat']].tail(period).rename(columns={"ds": "Date", "yhat": "Predicted Price"})
forecast_tail["Predicted Price"] = forecast_tail["Predicted Price"].round(2)
st.subheader(f"Prediction for the next {period} days (in USD)")
st.dataframe(forecast_tail)

# Summary in English
start_price = forecast_tail["Predicted Price"].iloc[0]
end_price = forecast_tail["Predicted Price"].iloc[-1]
change = end_price - start_price

st.subheader("AI Prediction Summary:")
if change > 1:
    trend = "upward"
elif change < -1:
    trend = "downward"
else:
    trend = "flat"

st.markdown(f"According to AI prediction, considering the impact of recent global stock market trends on gold prices, "
            f"it is expected that gold prices will trend **{trend}**, from approximately **${start_price:.2f}** to **${end_price:.2f}** over the next {period} days.")
