from pandas_datareader import data
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

ticker = "AAL"
file_to_save = f"stock_market_data_{ticker}_yf.csv"

df = yf.download(ticker)
df.to_csv("stock_market_data_1.csv")
df = df.iloc[::-1]
df2 = pd.DataFrame(columns=['Date', 'Low', 'High', 'Close', 'Open'])

for index, row in df.iterrows():
    date = str(index)[:str(index).find(" ")]
    date_info = dt.datetime.strptime(date, "%Y-%m-%d")
    df2.loc[-1, :] = [date_info.date(), row.iloc[2], row.iloc[1], row.iloc[0], row.iloc[3]]
    df2.index += 1

df2.to_csv(file_to_save)

plt.figure(figsize=(18,9))
plt.plot(range(df2.shape[0]), (df2['Low']+df2['High'])/2.0)
plt.xticks(range(0, df2.shape[0], 500), df2['Date'].loc[::500], rotation=45)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Mid Price', fontsize=18)
plt.show()

high_prices = df.loc[:, 'High'].values
low_prices = df.loc[:, 'Low'].values
mid_prices = (high_prices + low_prices) / 2.0

train_data = mid_prices[:730]
test_data = mid_prices[730:]

scaler = MinMaxScaler()
train_data = train_data.reshape(-1, 1)
test_data = test_data.reshape(-1, 1)

smoothing_window_size = 20

for di in range(0, 710, smoothing_window_size):
    scaler.fit(train_data[di:di + smoothing_window_size, :])
    train_data[di:di + smoothing_window_size, :] = scaler.transform(train_data[di:di + smoothing_window_size, :])

scaler.fit(train_data[di + smoothing_window_size:, :])
train_data[di + smoothing_window_size:, :] = scaler.transform(train_data[di + smoothing_window_size:, :])

train_data = train_data.reshape(-1)
test_data = scaler.transform(test_data).reshape(-1)

EMA = 0.0
gamma = 0.1

for ti in range(730):
    EMA = gamma * train_data[ti] + (1 - gamma) * EMA
    train_data[ti] = EMA

all_mid_data = np.concatenate([train_data, test_data], axis=0)

window_size = 100
N = train_data.size
std_avg_predictions = []
std_avg_x = []
mse_errors = []

for pred_idx in range(window_size, N):
    if pred_idx >= N:
        date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
    else:
        date = df2.loc[pred_idx, 'Date']

    std_avg_predictions.append(np.mean(train_data[pred_idx - window_size:pred_idx]))
    mse_errors.append((std_avg_predictions[-1] - train_data[pred_idx] ** 2))
    std_avg_x.append(date)

print(f"MSE error for standard averaging: {0.5 * np.mean(mse_errors)}")
print(len(std_avg_predictions))

"""plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]), all_mid_data, color='b', label='True')
plt.plot(range(window_size, N), std_avg_predictions, color='orange', label='Prediction')
plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.legend(fontsize=18)
plt.show()"""

window_size = 100
N = train_data.size

run_avg_predictions = []
run_avg_x = []

mse_errors = []

running_mean = 0.0
run_avg_predictions.append(running_mean)

decay = 0.5

for pred_idx in range(1, N):
    running_mean = running_mean * decay + (1.0 - decay) * train_data[pred_idx - 1]
    run_avg_predictions.append(running_mean)
    mse_errors.append((run_avg_predictions[-1] - train_data[pred_idx]) ** 2)
    run_avg_x.append(date)

print(f'MSE error for EMA averaging: {0.5 * np.mean(mse_errors)}')

plt.figure(figsize=(18,9))
plt.plot(range(df2.shape[0]), all_mid_data, color='b', label='True')
plt.plot(range(0,N), run_avg_predictions, color='orange', label='Prediction')
plt.xlabel("Date")
plt.ylabel("Prices")
plt.legend(fontsize=18)
plt.show()