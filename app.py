import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from scipy.stats import boxcox
from matplotlib import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from matplotlib import colors

# Set the style for the plots
sns.set_style('darkgrid')

# Load the data (Make sure the path is correct)
df = pd.read_csv('Electric_Production.csv', parse_dates=['DATE'], index_col='DATE')
df.columns = ['value']

# Set up Streamlit page layout
st.set_page_config(layout="wide")

# Sidebar for navigation
st.sidebar.title("Select a Graph to View")
option = st.sidebar.radio(
    "Choose a section to view:",
    ["Rolling Statistics", "ADF Test", "Log Transformation", "Moving Average", "Exponential Decay Transformation",
     "Seasonality Decomposition", "Autocorrelation and PACF", "Persistence Model", "ARIMA Models", "MSE Comparison"]
)

# Rolling Statistics
if option == "Rolling Statistics":
    # Plotting rolling statistics
    rolling_mean = df.rolling(window=12).mean()
    rolling_std = df.rolling(window=12).std()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df, color='cornflowerblue', label='Original')
    ax.plot(rolling_mean, color='firebrick', label='Rolling Mean')
    ax.plot(rolling_std, color='limegreen', label='Rolling Std')
    ax.set_xlabel('Date', size=12)
    ax.set_ylabel('Electric Production', size=12)
    ax.set_title('Rolling Statistics', size=14)
    ax.legend()
    st.pyplot(fig)

# ADF Test
elif option == "ADF Test":
    def adfuller_test(ts, window=12):
        movingAverage = ts.rolling(window).mean()
        movingSTD = ts.rolling(window).std()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ts, color='cornflowerblue', label='Original')
        ax.plot(movingAverage, color='firebrick', label='Rolling Mean')
        ax.plot(movingSTD, color='limegreen', label='Rolling Std')
        ax.set_title('Rolling Statistics', size=14)
        ax.legend()
        st.pyplot(fig)

        adf = adfuller(ts, autolag='AIC')
        st.write(f'ADF Statistic: {round(adf[0], 3)}')
        st.write(f'p-value: {round(adf[1], 3)}')
        st.write("##################################")
        st.write('Critical Values:')
        for key, ts in adf[4].items():
            st.write(f'{key}: {round(ts, 3)}')
        st.write("##################################")

        if adf[0] > adf[4]["5%"]:
            st.write("ADF > Critical Values")
            st.write("Failed to reject null hypothesis, time series is non-stationary.")
        else:
            st.write("ADF < Critical Values")
            st.write("Reject null hypothesis, time series is stationary.")

    adfuller_test(df, window=12)

# Log Transformation
elif option == "Log Transformation":
    df_log_scaled = df.copy()
    df_log_scaled['value'] = boxcox(df_log_scaled['value'], lmbda=0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_log_scaled, color='cornflowerblue')
    ax.set_xlabel('Date', size=12)
    ax.set_ylabel('Electric Production', size=12)
    ax.set_title("After Logarithmic Transformation", size=14)
    st.pyplot(fig)

# Moving Average
elif option == "Moving Average":
    df_log_scaled = df.copy()
    df_log_scaled['value'] = boxcox(df_log_scaled['value'], lmbda=0.0)

    moving_avg = df_log_scaled.rolling(window=12).mean()
    df_log_scaled_ma = df_log_scaled - moving_avg
    df_log_scaled_ma.dropna(inplace=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_log_scaled_ma, color='cornflowerblue')
    ax.set_xlabel('Date', size=12)
    ax.set_ylabel('Electric Production', size=12)
    ax.set_title("After Moving Average", size=14)
    st.pyplot(fig)

# Exponential Decay Transformation
elif option == "Exponential Decay Transformation":
    df_log_scaled = df.copy()
    df_log_scaled['value'] = boxcox(df_log_scaled['value'], lmbda=0.0)

    moving_avg = df_log_scaled.rolling(window=12).mean()
    df_log_scaled_ma = df_log_scaled - moving_avg
    df_log_scaled_ma.dropna(inplace=True)

    df_log_scaled_ma_ed = df_log_scaled_ma.ewm(halflife=12, min_periods=0, adjust=True).mean()
    df_lsma_sub_df_lsma_ed = df_log_scaled_ma - df_log_scaled_ma_ed

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_lsma_sub_df_lsma_ed, color='cornflowerblue')
    ax.set_xlabel('Date', size=12)
    ax.set_ylabel('Electric Production', size=12)
    ax.set_title("After Exponential Decay Transformation", size=14)
    st.pyplot(fig)

# Seasonality Decomposition
elif option == "Seasonality Decomposition":
    df_log_scaled = df.copy()
    df_log_scaled['value'] = boxcox(df_log_scaled['value'], lmbda=0.0)

    moving_avg = df_log_scaled.rolling(window=12).mean()
    df_log_scaled_ma = df_log_scaled - moving_avg
    df_log_scaled_ma.dropna(inplace=True)

    df_seasonal_decompose = seasonal_decompose(df_log_scaled_ma, model='additive')
    fig = df_seasonal_decompose.plot()
    st.pyplot(fig)

# Autocorrelation and Partial Autocorrelation
elif option == "Autocorrelation and PACF":
    df_log_scaled = df.copy()
    df_log_scaled['value'] = boxcox(df_log_scaled['value'], lmbda=0.0)

    moving_avg = df_log_scaled.rolling(window=12).mean()
    df_log_scaled_ma = df_log_scaled - moving_avg
    df_log_scaled_ma.dropna(inplace=True)

    auto_c_f = acf(df_log_scaled_ma, nlags=20)
    partial_auto_c_f = pacf(df_log_scaled_ma, nlags=20, method='ols')

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # ACF plot
    axs[0].plot(auto_c_f)
    axs[0].axhline(y=0, linestyle='--', color='limegreen')
    axs[0].axhline(y=-1.96/np.sqrt(len(df_log_scaled_ma)), linestyle='--', color='firebrick')
    axs[0].axhline(y=1.96/np.sqrt(len(df_log_scaled_ma)), linestyle='--', color='firebrick')
    axs[0].set_title('Autocorrelation Function', size=14)

    # PACF plot
    axs[1].plot(partial_auto_c_f)
    axs[1].axhline(y=0, linestyle='--', color='limegreen')
    axs[1].axhline(y=-1.96/np.sqrt(len(df_log_scaled_ma)), linestyle='--', color='firebrick')
    axs[1].axhline(y=1.96/np.sqrt(len(df_log_scaled_ma)), linestyle='--', color='firebrick')
    axs[1].set_title('Partial Autocorrelation Function', size=14)

    plt.tight_layout()
    st.pyplot(fig)

# Persistence Model
elif option == "Persistence Model":
    df_log_scaled = df.copy()
    df_log_scaled['value'] = boxcox(df_log_scaled['value'], lmbda=0.0)

    train_size = int(len(df_log_scaled) * 0.66)
    train, test = df_log_scaled[0:train_size], df_log_scaled[train_size:]
    history = train['value'].tolist()
    predictions = []

    for t in range(len(test)):
        yhat = history[-1]  # last observed value
        predictions.append(yhat)
        history.append(test['value'][t])

    mse = mean_squared_error(test['value'], predictions)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(test.index, test['value'], label='Actual', color='cornflowerblue')
    ax.plot(test.index, predictions, label='Predicted', color='orange')
    ax.set_title('Persistence Model Forecast', size=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Electric Production')
    ax.legend()
    st.pyplot(fig)
    st.write(f"Mean Squared Error of Persistence Model: **{round(mse, 4)}**")


# ARIMA Models
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
    ax.set_title('ARIMA Model Forecast', size=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Electric Production')
    ax.legend()
    st.pyplot(fig)
    st.write(f"Mean Squared Error of ARIMA Model: **{round(mse, 4)}**")


# MSE Comparison
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

    st.write("### MSE Comparison of Models")
    st.write(f"Persistence Model MSE: **{round(persistence_mse, 4)}**")
    st.write(f"ARIMA Model MSE: **{round(arima_mse, 4)}**")

    # Bar Chart
    fig, ax = plt.subplots(figsize=(6, 4))
    models = ['Persistence', 'ARIMA']
    mses = [persistence_mse, arima_mse]
    sns.barplot(x=models, y=mses, palette='viridis', ax=ax)
    ax.set_title("Mean Squared Error Comparison")
    ax.set_ylabel("MSE")
    st.pyplot(fig)

