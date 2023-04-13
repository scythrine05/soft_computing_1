import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

def BackPropAlgo(u):
    # Load stock market data from file
    data = pd.read_csv('...\stock_market_train.csv')
    Open = data['Open'].values
    High = data['High'].values
    Low = data['Low'].values
    Close = data['Close'].values

    # Compute technical indicators
    SMA_10 = np.convolve(Open, np.ones(10) / 10, mode='valid')
    SMA_50 = np.convolve(Open, np.ones(50) / 50, mode='valid')
    EMA_10 = pd.Series(Open).ewm(span=10, min_periods=10).mean().values
    EMA_50 = pd.Series(Open).ewm(span=50, min_periods=50).mean().values

    # Combine inputs into a single array
    Input = np.array([Open, High, Low, SMA_10, EMA_10, SMA_50, EMA_50]).T

    # Create and train neural network
    net = MLPRegressor(hidden_layer_sizes=(abs(int(u)), 1), activation='identity', solver='lbfgs')
    net.fit(Input, Close)

    # Calculate performance metric
    t = net.predict(Input)
    perf = np.sqrt(np.mean((Close - t) ** 2))

    if perf < 0:
        perf = 1e20

    return perf