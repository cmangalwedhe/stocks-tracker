import base64

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from io import BytesIO

class StockPriceLSTM:
    def __init__(self, ticker, sequence_length=60):
        self.ticker = ticker
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.data = None

    def fetch_data(self, years=5):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)

        try:
            df = yf.download(self.ticker, start=start_date, end=end_date)
            if df.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")

            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['RSI'] = self._calculate_rsi(df['Close'])
            df['Daily_Return'] = df['Close'].pct_change()

            df.dropna(inplace=True)
            self.data = df
            return df

        except Exception as e:
            raise Exception(f"Error fetching data: {str(e)}")

    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def prepare_data(self, test_size=0.2):
        if self.data is None:
            raise ValueError("No data available. Call fetch_data() first.")

        features = ['Close', 'MA5', 'MA20', 'RSI', 'Daily_Return']
        dataset = self.data[features].values

        scaled_data = self.scaler.fit_transform(dataset)

        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i - self.sequence_length:i])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)

        train_size = int(len(X) * (1 - test_size))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        return X_train, y_train, X_test, y_test

    def build_model(self):
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(self.sequence_length, 5)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='mean_squared_error')

        self.model = model
        return model

    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        if self.model is None:
            self.build_model()

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )

        return history

    def predict_future(self, days=30):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        features = ['Close', 'MA5', 'MA20', 'RSI', 'Daily_Return']
        last_sequence = self.data[features].values[-self.sequence_length:]
        last_sequence = self.scaler.transform(last_sequence)

        predictions = []
        current_sequence = last_sequence.copy()

        for _ in range(days):
            current_sequence_reshaped = current_sequence.reshape((1, self.sequence_length, len(features)))
            next_day = self.model.predict(current_sequence_reshaped, verbose=0)
            predictions.append(next_day[0, 0])

            current_sequence = np.roll(current_sequence, -1, axis=0)
            new_row = np.zeros(len(features))
            new_row[0] = next_day[0, 0]
            current_sequence[-1] = new_row

        predictions_array = np.array(predictions).reshape(-1, 1)
        predictions_transformed = self.scaler.inverse_transform(
            np.hstack([predictions_array, np.zeros((len(predictions), len(features) - 1))])
        )[:, 0]

        return predictions_transformed

    def plot_predictions(self, predictions, days_to_show=60):
        plt.figure(figsize=(15, 7))

        historical = self.data['Close'].values[-days_to_show:]
        plt.plot(range(days_to_show), historical,
                 label='Historical Prices', color='blue')

        plt.plot(range(days_to_show - 1, days_to_show + len(predictions)),
                 np.append(historical[-1], predictions),
                 label='Predicted Prices', color='red', linestyle='--')

        plt.title(f'{self.ticker} Stock Price Prediction')
        plt.xlabel('Days')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        encoded_image = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()

        return encoded_image

def predict_stock_prices(ticker, days_to_predict=30, sequence_length=60):
    try:
        model = StockPriceLSTM(ticker, sequence_length=sequence_length)

        print(f"Fetching historical data for {ticker}...")
        model.fetch_data()

        print("Preparing data for training...")
        X_train, y_train, X_test, y_test = model.prepare_data()

        print("Training LSTM model...")
        model.train(X_train, y_train)

        print(f"Predicting next {days_to_predict} days...")
        predictions = model.predict_future(days=days_to_predict)

        plot = model.plot_predictions(predictions)
        return plot, predictions

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None