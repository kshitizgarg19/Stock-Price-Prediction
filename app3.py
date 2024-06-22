import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Define the stock tickers and company names for easy selection
us_companies = {
    'Apple (AAPL)': 'AAPL',
    'Microsoft (MSFT)': 'MSFT',
    'Amazon (AMZN)': 'AMZN',
    'Google (GOOGL)': 'GOOGL',
    'Tesla (TSLA)': 'TSLA'
}

indian_companies = {
    'Infosys (INFY)': 'INFY',
    'Reliance Industries (RELIANCE.NS)': 'RELIANCE.NS',
    'Tata Consultancy Services (TCS.NS)': 'TCS.NS',
    'HDFC Bank (HDFCBANK.NS)': 'HDFCBANK.NS',
    'ICICI Bank (ICICIBANK.NS)': 'ICICIBANK.NS',
    'Sensex': '^BSESN',
    'Nifty 50': '^NSEI',
    'Bank Nifty': '^NSEBANK'
}

# Streamlit app
st.title('Stock Price Prediction')

# User input for selecting a market
market = st.radio('Select a market:', ('US', 'India'))

if market == 'US':
    companies = us_companies
else:
    companies = indian_companies

# User input for selecting a company
selected_company = st.selectbox('Select a company:', list(companies.keys()))

# Fetching the data
ticker = companies[selected_company]
start_date = '2020-01-01'
end_date = '2024-06-22'  # Current date
df = yf.download(ticker, start=start_date, end=end_date)

# Preprocessing the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Creating training data
X_train = []
y_train = []
for i in range(100, len(df)):
    X_train.append(scaled_data[i-100:i, 0])
    y_train.append(scaled_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Creating the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
X_test = scaled_data[len(df)-100:len(df), 0]
X_test = X_test.reshape(1, -1)
predicted_price = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))

# Plotting the live prices
st.subheader('Recent Stock Prices')
st.write(df.tail())

# Plotting the historical closing price vs. time
st.subheader('Historical Closing Price vs. Time')
fig, ax = plt.subplots()
ax.plot(df.index, df['Close'], label='Closing Price')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Historical Closing Price vs. Time')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

# Plotting the MA100 and MA200
df['MA100'] = df['Close'].rolling(window=100).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()
ax.plot(df.index, df['MA100'], label='MA100')
ax.plot(df.index, df['MA200'], label='MA200')
ax.legend()

st.pyplot(fig)

# Displaying the predicted price
st.subheader('Predicted Price')
st.write(f"Predicted Price: {predicted_price[0][0]}")

# Plotting the predicted price curve
fig, ax = plt.subplots()
ax.plot(df.index[-100:], df['Close'][-100:], label='Historical Price')
ax.plot(df.index[-1], predicted_price[0][0], 'ro', label='Predicted Price')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Predicted Price Curve')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
st.pyplot(fig)
