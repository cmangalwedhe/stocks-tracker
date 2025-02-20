from pandas_datareader import data
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

import yfinance as yf

# loading data from yahoo finance and storing in a pandas dataframe which is easy to read to non-programmers
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


# calculate multiple prices and use as different parameters in deciding which is most accurate
high_prices = df.loc[:, 'High'].values
low_prices = df.loc[:, 'Low'].values
close_prices = df.loc[:, 'Close'].values
mid_prices = (high_prices + low_prices) / 2.0


# split data into training and testing lists and
scaler = MinMaxScaler(feature_range=(0,1))
split = int(.45 * len(close_prices))
training_data = close_prices[:split]
testing_data = close_prices[split:]


# scale data using mean and standard deviation (helps to standardize data)
data_training_list = scaler.fit_transform(training_data)


x_train = []
y_train = []

# create different roll-outs of data for model training (batches of 100 days) and
for i in range(0, data_training_list.shape[0] - 100):
    x_train.append(data_training_list[i:i+100])
    y_train.append(data_training_list[i, 0])


x_train, y_train = np.array(x_train), np.array(y_train)

model = tf.keras.Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError()])
model.fit(x_train, y_train, epochs=100)

past_100_days = pd.DataFrame(training_data[-100:])
test_df = pd.DataFrame(testing_data)
final_df = past_100_days._append(test_df, ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(0, input_data.shape[0]-100):
    x_test.append(input_data[i:i+100])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_pred = model.predict(x_test)

scale_factor = 1/scaler.scale_[0]
y_pred *= scale_factor
y_test *= scale_factor

plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = "Original Price")
plt.plot(y_pred, 'r', label = "Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

mae = mean_absolute_error(y_test, y_pred)
mae_percentage = (mae / np.mean(y_test)) * 100
print(f"Mean Absolute Error on test set: {mae_percentage:.2f}%")
