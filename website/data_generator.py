import numpy as np
import os
import urllib.request, json
import datetime as dt
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler


API_KEY = "I57LV5Y7UJALBAFS"
ticker = "AAPL"
url_string = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={API_KEY}"

file_to_save = f"stock_market_data_{ticker}.csv"

if not os.path.exists(file_to_save):
    with urllib.request.urlopen(url_string) as url:
        data = json.loads(url.read().decode())
        data = data['Time Series (Daily)']
        df = pd.DataFrame(columns=['Date', 'Low', 'High', 'Close', 'Open'])

        for k, v in data.items():
            date = dt.datetime.strptime(k, "%Y-%m-%d")
            data_row = [date.date(), float(v['3. low']), float(v['2. high']), float(v['4. close']), float(v['1. open'])]
            df.loc[-1,:] = data_row
            df.index += 1

        print(f"Data saved to: {file_to_save}")
        df.to_csv(file_to_save)
else:
    print("File already exists. Loading data from CSV.")


class DataGeneratorSeq:
    def __init__(self, prices, batch_size, num_unroll):
        self.prices = prices
        self.prices_length = len(self.prices) - num_unroll
        self.batch_size = batch_size
        self.num_unroll = num_unroll
        self.segments = self.prices_length // self.batch_size
        self.cursor = [offset * self.segments for offset in range(self.batch_size)]

    def next_batch(self):
        batch_data = np.zeros(self.batch_size, dtype=np.float32)
        batch_labels = np.zeros(self.batch_size, dtype=np.float32)

        for batch in range(self.batch_size):
            if self.cursor[batch] + 1 >= self.prices_length:
                self.cursor[batch] = np.random.randint(0, (batch+1) * self.segments)

            batch_data[batch] = self.prices[self.cursor[batch]]
            batch_labels[batch] = self.prices[self.cursor[batch] + np.random.randint(0, 5)]

            self.cursor[batch] = (self.cursor[batch] + 1) % self.prices_length

        return batch_data, batch_labels

    def unroll_batches(self):
        unroll_data, unroll_labels = [], []
        initial_data, initial_label = None, None

        for ui in range(self.num_unroll):
            data, labels = self.next_batch()
            unroll_data.append(data)
            unroll_labels.append(labels)

        return unroll_data, unroll_labels

    def reset_indices(self):
        for b in range(self.batch_size):
            self.cursor[b] = np.random.randint(0, min((b+1)*self.segments, self.prices_length-1))


high_prices = df.loc[:,'High'].as_matrix()
low_prices = df.loc[:,'Low'].as_matrix()
mid_prices = (high_prices + low_prices) / 2.0

train_data = mid_prices[:11000]
test_data = mid_prices[11000:]

scaler = MinMaxScaler()
train_data = train_data.reshape(-1, 1)
test_data = test_data.reshape(-1, 1)

smoothing_data_size = 2500

for di in range(0, 1000, smoothing_data_size):
    scaler.fit(train_data[di:di+smoothing_data_size,:])
    train_data[di:di+smoothing_data_size,:] = scaler.transform(train_data[di:di+smoothing_data_size,:])

scaler.fit(train_data[di + smoothing_data_size:,:])
train_data[di+smoothing_data_size:,:] = scaler.transform(train_data)

train_data = train_data.reshape(-1)
test_data = scaler.transform(test_data).reshape(-1)

