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

# Create a
