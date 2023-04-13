import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import argrelextrema
from sklearn.neural_network import MLPRegressor

# Opening the training test data
data_train = np.genfromtxt('.../stock_market_train.csv', delimiter=',')[1:]
Open = data_train[:, 1]
High = data_train[:, 2]
Low = data_train[:, 3]
Close = data_train[:, 4]

# Simple Moving Average for 10 and 50 days
SMA_10 = np.convolve(Open, np.ones((10,))/10, mode='valid')
SMA_50 = np.convolve(Open, np.ones((50,))/50, mode='valid')

# Exponential Moving Average for 10 and 50 days
EMA_10 = np.convolve(Open, np.ones((10,))/10*(2/(1+10)), mode='valid')
EMA_50 = np.convolve(Open, np.ones((50,))/50*(2/(1+50)), mode='valid')

# Input vector of the input variables
Input = np.vstack((Open, High, Low, SMA_10, EMA_10, SMA_50, EMA_50)).T

# Construction of feed-forward neural network
net = MLPRegressor(hidden_layer_sizes=(7,), activation='identity', solver='adam',
                   learning_rate_init=0.001, max_iter=8000, tol=1e-5)

# Using full data to train the neural network
net.fit(Input, Close)
t = net.predict(Input)

# Evaluating the performance of the neural network - using mse as the measuring standard
perf = np.mean((t - Close) ** 2)
print('MSE: ', perf)

# Plot generation of the market values
x = np.arange(Close.size)
plt.plot(x, Close, x, Open, x, High, x, Low)
plt.legend(('Close', 'Open', 'High', 'Low'), loc='upper left')
plt.title('Training Data')
plt.xlabel('Data Points')
plt.ylabel('Stock Market Value')
plt.show()

# Opening sample test data
data_test = np.genfromtxt('.../stock_market_test_final.csv', delimiter=',')[1:]
Open_t = data_test[:, 1]
High_t = data_test[:, 2]
Low_t = data_test[:, 3]
Close_t = data_test[:, 4]

SMA_10_t = np.convolve(Open_t, np.ones((10,))/10, mode='valid')
SMA_50_t = np.convolve(Open_t, np.ones((50,))/50, mode='valid')
EMA_10_t = np.convolve(Open_t, np.ones((10,))/10*(2/(1+10)), mode='valid')
EMA_50_t = np.convolve(Open_t, np.ones((50,))/50*(2/(1+50)), mode='valid')

Input_t = np.vstack((Open_t, High_t, Low_t, SMA_10_t, EMA_10_t, SMA_50_t, EMA_50_t)).T

# Plotting the final output graph
answer = net.predict(Input_t)

x = np.arange(Close_t.size)
plt.plot(x, Close_t, x, answer)
plt.legend(('Actual Value', 'Predicted Value'), loc='upper left')
plt.title('Stock Market Prediction using Neural Networks')
plt.xlabel('Data Points')
plt.ylabel('Closing Stock Market Value')
plt.show()
