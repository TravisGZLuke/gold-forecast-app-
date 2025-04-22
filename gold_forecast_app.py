import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Set up the page configuration
st.set_page_config(page_title="Gold Price Prediction App", layout="centered")
st.title("Gold Price Prediction (Considering Global Stock Market Impact)")
st.markdown("This AI model uses linear regression to predict gold prices, considering the impact of global stock market trends.")

# User selects forecast period (days)
period = st.selectbox("Choose forecast period", [7, 14, 30])

# Fetch gold and global stock market data
@st.cache_data
def get_all_data():
    # Fetch Gold ETF (GLD) data
    gold = yf.download("GLD", start="2015-01-01")[['Close']].rename(columns={"Close": "gold"})
    
    # Fetch S&P500 index data
    sp500 = yf.download("^GSPC", start="2015-01-01")[['Close']].rename(columns={"Close": "sp500"})
    
    # Fetch Nikkei 225 index data
    nikkei = yf.download("^N225", start="2015-01-01")[['Close']].rename(columns={"Close": "nikkei"})
    
    # Fetch DAX index data
    dax = yf.download("^GDAXI", start="2015-01-01")[['Close']].rename(columns={"Close": "dax"})

    # Merge data using outer join to avoid losing any columns
    df = gold.join([sp500, nikkei, dax], how='outer')  # Use 'outer' join to ensure no data loss
    
    # Reset index and rename columns for future processing
    df = df.reset_index().rename(columns={"Date": "ds"})
    
    # Rename 'gold' column to 'y' because we need it as the target column for regression
    df = df.rename(columns={"gold": "y"})
    
    # Return the merged dataframe
    return df

df = get_all_data()

# Display historical gold prices with stock market trends
st.subheader("Historical Gold Prices vs Global Stock Markets")
st.line_chart(df.set_index("ds")[["y", "sp500", "nikkei", "dax"]])

# Create a linear regression model
model = LinearRegression()

# Prepare the data for regression analysis
X = df[["sp500", "nikkei", "dax"]]  # Features: stock market data
y = df["y"]  # Target variable: gold price

# Train the regression model
model.fit(X, y)

# Fetch the latest stock market data (assuming stock market data won't change dramatically)
latest_sp500 = df["sp500"].iloc[-1]
latest_nikkei = df["nikkei"].iloc[-1]
latest_dax = df["dax"].iloc[-1]

# Create a future dataframe with stock market data filled in
future_X = pd.DataFrame({
    "sp500": [latest_sp500] * period,
    "nikkei": [latest_nikkei] * period,
    "dax": [latest_dax] * period
})

# Use the regression model to predict gold prices
forecast_y = model.predict(future_X)

# Display prediction chart
forecast_dates = pd.date_range(df["ds"].iloc[-1], periods=period + 1, freq='D')[1:]

# Create a DataFrame to store the prediction results
forecast_df = pd.DataFrame({
    "Date": forecast_dates,
    "Predicted Price": forecast_y
})

st.subheader(f"Prediction for the next {period} days (in USD)")
st.dataframe(forecast_df)

# Display prediction chart
st.subheader("Prediction Chart")
plt.figure(figsize=(10, 6))
plt.plot(df["ds"], df["y"], label="Historical Gold Price", color="blue")
plt.plot(forecast_df["Date"], forecast_df["Predicted Price"], label=f"Predicted Gold Price ({period} days)", color="red")
plt.xlabel("Date")
plt.ylabel("Gold Price (USD)")
plt.title("Gold Price Prediction with Stock Market Impact")
plt.legend()
st.pyplot(plt)
