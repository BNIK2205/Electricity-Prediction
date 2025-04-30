import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from scipy.stats import boxcox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf

# Set Streamlit config
st.set_page_config(page_title="Electricity Forecasting", layout="wide")

# Title and Description
st.title("⚡ Electricity Production Forecasting")
st.markdown("""
Welcome to the interactive time series analysis dashboard for **US Electricity Production**.
Use the sidebar to explore data transformations, model performance, and forecasting methods.
""")

# Sidebar
with st.sidebar:
    st.header("📊 Analysis Menu")
    option = st.radio(
        "Choose a section:",
        ["Rolling Statistics", "ADF Test", "Log Transformation", "Moving Average",
         "Exponential Decay Transformation", "Seasonality Decomposition",
         "Autocorrelation and PACF", "Persistence Model", "ARIMA Models", "MSE Comparison"]
    )
    st.markdown("---")
   # st.markdown("👨‍💻 Built by NikhilBandi")

# Load Data
try:
    df = pd.read_csv('Electric_Production.csv', parse_dates=['DATE'], index_col='DATE')
    df.columns = ['value']
except FileNotFoundError:
    st.error("The file 'Electric_Production.csv' was not found. Please upload it to continue.")
    st.stop()

sns.set_style('darkgrid')

# Rolling Statistics
if option == "Rolling Statistics":
    rolling_mean = df.rolling(window=12).mean()
    rolling_std = df.rolling(window=12).std()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df, label='Original', color='cornflowerblue')
    ax.plot(rolling_mean, label='Rolling Mean', color='firebrick')
    ax.plot(rolling_std, label='Rolling Std', color='limegreen')
    ax.set_title('Rolling Statistics')
    ax.legend()
    st.pyplot(fig)

elif option == "ADF Test":
    def adfuller_test(ts, window=12):
        movingAverage = ts.rolling(window).mean()
        movingSTD = ts.rolling(window).std()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ts, label='Original', color='cornflowerblue')
        ax.plot(movingAverage, label='Rolling Mean', color='firebrick')
        ax.plot(movingSTD, label='Rolling Std', color='limegreen')
        ax.set_title('Rolling Statistics')
        ax.legend()
        st.pyplot(fig)

        adf = adfuller(ts, autolag='AIC')
        with st.expander("🔍 ADF Test Results"):
            st.write(f"**ADF Statistic:** {round(adf[0], 3)}")
            st.write(f"**p-value:** {round(adf[1], 3)}")
            st.write("**Critical Values:**")
            for key, ts in adf[4].items():
                st.write(f"- {key}: {round(ts, 3)}")
            if adf[0] > adf[4]["5%"]:
                st.error("Time series is non-stationary.")
            else:
                st.success("Time series is stationary.")

    adfuller_test(df['value'])

elif option == "Log Transformation":
    df_log_scaled = df.copy()
    df_log_scaled['value'] = boxcox(df_log_scaled['value'], lmbda=0.0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_log_scaled, color='cornflowerblue')
    ax.set_title("After Logarithmic Transformation")
    st.pyplot(fig)

elif option == "Moving Average":
    df_log_scaled = df.copy()
    df_log_scaled['value'] = boxcox(df_log_scaled['value'], lmbda=0.0)
    moving_avg = df_log_scaled.rolling(window=12).mean()
    df_log_scaled_ma = df_log_scaled - moving_avg
    df_log_scaled_ma.dropna(inplace=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_log_scaled_ma, color='cornflowerblue')
    ax.set_title("After Moving Average")
    st.pyplot(fig)

elif option == "Exponential Decay Transformation":
    df_log_scaled = df.copy()
    df_log_scaled['value'] = boxcox(df_log_scaled['value'], lmbda=0.0)
    moving_avg = df_log_scaled.rolling(window=12).mean()
    df_log_scaled_ma = df_log_scaled - moving_avg
    df_log_scaled_ma.dropna(inplace=True)
    df_log_scaled_ma_ed = df_log_scaled_ma.ewm(halflife=12).mean()
    diff = df_log_scaled_ma - df_log_scaled_ma_ed
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(diff, color='cornflowerblue')
    ax.set_title("After Exponential Decay Transformation")
    st.pyplot(fig)

elif option == "Seasonality Decomposition":
    df_log_scaled = df.copy()
    df_log_scaled['value'] = boxcox(df_log_scaled['value'], lmbda=0.0)
    moving_avg = df_log_scaled.rolling(window=12).mean()
    df_log_scaled_ma = df_log_scaled - moving_avg
    df_log_scaled_ma.dropna(inplace=True)
    df_seasonal_decompose = seasonal_decompose(df_log_scaled_ma, model='additive')
    fig = df_seasonal_decompose.plot()
    st.pyplot(fig)

elif option == "Autocorrelation and PACF":
    df_log_scaled = df.copy()
    df_log_scaled['value'] = boxcox(df_log_scaled['value'], lmbda=0.0)
    moving_avg = df_log_scaled.rolling(window=12).mean()
    df_log_scaled_ma = df_log_scaled - moving_avg
    df_log_scaled_ma.dropna(inplace=True)
    auto_c_f = acf(df_log_scaled_ma, nlags=20)
    partial_auto_c_f = pacf(df_log_scaled_ma, nlags=20)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].stem(auto_c_f)
    axs[0].set_title('ACF')
    axs[1].stem(partial_auto_c_f)
    axs[1].set_title('PACF')
    st.pyplot(fig)

elif option == "Persistence Model":
    df_log_scaled = df.copy()
    df_log_scaled['value'] = boxcox(df_log_scaled['value'], lmbda=0.0)
    train_size = int(len(df_log_scaled) * 0.66)
    train, test = df_log_scaled[0:train_size], df_log_scaled[train_size:]
    history = train['value'].tolist()
    predictions = []
    for t in range(len(test)):
        yhat = history[-1]
        predictions.append(yhat)
        history.append(test['value'][t])
    mse = mean_squared_error(test['value'], predictions)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(test.index, test['value'], label='Actual', color='cornflowerblue')
    ax.plot(test.index, predictions, label='Predicted', color='orange')
    ax.set_title('Persistence Model Forecast')
    ax.legend()
    st.pyplot(fig)
    st.metric("Mean Squared Error", round(mse, 4))

elif option == "ARIMA Models":
    df_log_scaled = df.copy()
    df_log_scaled['value'] = boxcox(df_log_scaled['value'], lmbda=0.0)
    train_size = int(len(df_log_scaled) * 0.66)
    train, test = df_log_scaled[0:train_size], df_log_scaled[train_size:]
    model = ARIMA(train, order=(2, 1, 2))
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test))
    mse = mean_squared_error(test['value'], predictions)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(test.index, test['value'], label='Actual', color='cornflowerblue')
    ax.plot(test.index, predictions, label='ARIMA Predicted', color='orange')
    ax.set_title('ARIMA Forecast')
    ax.legend()
    st.pyplot(fig)
    st.metric("Mean Squared Error", round(mse, 4))

elif option == "MSE Comparison":
    df_log_scaled = df.copy()
    df_log_scaled['value'] = boxcox(df_log_scaled['value'], lmbda=0.0)
    train_size = int(len(df_log_scaled) * 0.66)
    train, test = df_log_scaled[0:train_size], df_log_scaled[train_size:]
    # Persistence Model
    history = train['value'].tolist()
    persistence_preds = []
    for t in range(len(test)):
        yhat = history[-1]
        persistence_preds.append(yhat)
        history.append(test['value'][t])
    persistence_mse = mean_squared_error(test['value'], persistence_preds)
    # ARIMA Model
    model = ARIMA(train, order=(2, 1, 2))
    model_fit = model.fit()
    arima_preds = model_fit.forecast(steps=len(test))
    arima_mse = mean_squared_error(test['value'], arima_preds)
    st.subheader("📉 MSE Comparison")
    st.write(f"**Persistence Model MSE:** {round(persistence_mse, 4)}")
    st.write(f"**ARIMA Model MSE:** {round(arima_mse, 4)}")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=['Persistence', 'ARIMA'], y=[persistence_mse, arima_mse], palette='viridis', ax=ax)
    ax.set_title("Mean Squared Error Comparison")
    ax.set_ylabel("MSE")
    st.pyplot(fig)
