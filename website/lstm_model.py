import keras.src.callbacks
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings

warnings.filterwarnings('ignore')


class StockPricePredictor:
    def __init__(self, ticker, prediction_days=30):
        """
        Initialize the stock price predictor.

        Parameters:
        ticker (str): Stock ticker symbol
        prediction_days (int): Number of days to predict into the future
        """
        self.ticker = ticker
        self.prediction_days = prediction_days
        self.data = None
        self.features = None
        self.models = {}
        self.model_predictions = {}
        self.ensemble_weights = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def fetch_data(self, years=5):
        """
        Fetch historical stock data and market indicators.

        Parameters:
        years (int): Number of years of historical data to fetch

        Returns:
        pandas.DataFrame: Processed data ready for feature engineering
        """
        # Calculate start date based on years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)

        # Fetch the stock data
        stock_data = yf.download(self.ticker, start=start_date, end=end_date)

        # Fetch market index (S&P 500) as a reference
        sp500 = yf.download('^GSPC', start=start_date, end=end_date)

        # Also get the sector ETF if available (example for tech is XLK)
        # You would need to map tickers to their sector ETFs for better accuracy
        sector_mapping = {
            'AAPL': 'XLK', 'MSFT': 'XLK', 'GOOGL': 'XLC', 'AMZN': 'XLY', 'META': 'XLC',
            'TSLA': 'XLY', 'JPM': 'XLF', 'V': 'XLF', 'UNH': 'XLV', 'JNJ': 'XLV',
            'PG': 'XLP', 'HD': 'XLY', 'CVX': 'XLE', 'XOM': 'XLE', 'BAC': 'XLF'
        }

        sector_etf = sector_mapping.get(self.ticker, 'XLK')  # Default to XLK for tech
        try:
            sector_data = yf.download(sector_etf, start=start_date, end=end_date)
            has_sector_data = True
        except:
            has_sector_data = False

        # Ensure we have data
        if stock_data.empty:
            raise ValueError(f"No data found for ticker {self.ticker}")

        # Copy the stock data to avoid modifying the original
        df = stock_data.copy()

        # Add S&P 500 data as features
        if not sp500.empty:
            df['SP500_Close'] = sp500['Close']
            df['Market_Return'] = sp500['Close'].pct_change()

        # Add sector data if available
        if has_sector_data and not sector_data.empty:
            df[f'{sector_etf}_Close'] = sector_data['Close']
            df[f'Sector_Return'] = sector_data['Close'].pct_change()

        # Drop rows with NaN values
        df.dropna(inplace=True)

        # Store the data
        self.data = df

        return df

    def create_features(self):
        """
        Create features for the model.

        Returns:
        pandas.DataFrame: DataFrame with features
        """
        if self.data is None:
            raise ValueError("Data not fetched. Call fetch_data() first.")

        df = self.data.copy()

        # Basic price features
        df['Return'] = df['Close'].pct_change()
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()

        # Volatility indicators
        df['Std5'] = df['Close'].rolling(window=5).std()
        df['Std20'] = df['Close'].rolling(window=20).std()

        # Trading indicators
        df['RSI'] = self.calculate_rsi(df['Close'], periods=14)

        # MACD
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal_Line']

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']

        # Volume indicators
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_Ratio'] = df['Volume'][ticker] / df['Volume_MA5']

        # Price changes
        df['Price_Change'] = df['Close'].diff()
        df['Price_Change_Pct'] = df['Close'].pct_change() * 100

        # Momentum
        df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1

        # Rate of Change
        df['ROC_5'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 100
        df['ROC_10'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100

        # Lagged features for time series analysis
        for i in range(1, 6):
            df[f'Close_Lag_{i}'] = df['Close'].shift(i)
            df[f'Return_Lag_{i}'] = df['Return'].shift(i)

        # Day of week, month, etc. (calendar effects)
        df['DayOfWeek'] = pd.to_datetime(df.index).dayofweek
        df['Month'] = pd.to_datetime(df.index).month
        df['Quarter'] = pd.to_datetime(df.index).quarter

        # Relative strength compared to market
        if 'Market_Return' in df.columns:
            df['RS_Market'] = df['Return'] / df['Market_Return']

        # Relative strength compared to sector
        if 'Sector_Return' in df.columns:
            df['RS_Sector'] = df['Return'] / df['Sector_Return']

        # Drop rows with NaN values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # Store features
        self.features = df
        print(self.features.columns)
        return df

    def calculate_rsi(self, series, periods=14):
        """Calculate the RSI indicator."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=periods).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def prepare_data_for_models(self, test_size=0.2, target_col='Close'):
        """
        Prepare data for different model types.

        Parameters:
        test_size (float): Proportion of data to use for testing
        target_col (str): Column to predict

        Returns:
        dict: Dictionary of datasets for different models
        """
        if self.features is None:
            raise ValueError("Features not created. Call create_features() first.")

        data = self.features.copy()

        # Define features and target
        X = data.drop([target_col, 'Open', 'High', 'Low'], axis=1)
        y = data[target_col]

        for col in X.columns:
            for j in X[col]:
                if not np.isfinite(j):
                    print(f"col: {col} - val:{X[col][j]}")

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        # Scale data for neural network models
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Prepare data for LSTM (3D shape: [samples, timesteps, features])
        timesteps = 10  # Number of previous time steps to use
        X_train_lstm, y_train_lstm = self.create_sequences(X_train_scaled, y_train, timesteps)
        X_test_lstm, y_test_lstm = self.create_sequences(X_test_scaled, y_test, timesteps)

        return {
            'standard': (X_train, X_test, y_train, y_test),
            'scaled': (X_train_scaled, X_test_scaled, y_train, y_test),
            'lstm': (X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm)
        }

    def create_sequences(self, X, y, timesteps):
        """Create sequences for LSTM model."""
        X_seq, y_seq = [], []
        for i in range(len(X) - timesteps):
            X_seq.append(X[i:i + timesteps])
            y_seq.append(y.iloc[i + timesteps])
        return np.array(X_seq), np.array(y_seq)

    def build_models(self, datasets):
        """
        Build various predictive models.

        Parameters:
        datasets (dict): Dictionary of datasets for different models

        Returns:
        dict: Dictionary of trained models
        """
        models = {}

        # Unpack datasets
        X_train, X_test, y_train, y_test = datasets['standard']
        X_train_scaled, X_test_scaled, _, _ = datasets['scaled']
        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = datasets['lstm']

        # 1. Random Forest
        print("Training Random Forest model...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        models['random_forest'] = rf_model

        # 2. Gradient Boosting
        print("Training Gradient Boosting model...")
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train, y_train)
        models['gradient_boosting'] = gb_model

        # 3. SARIMAX (time series model)
        print("Training SARIMAX model...")
        try:
            sarimax_model = SARIMAX(
                y_train,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 5)
            )
            sarimax_results = sarimax_model.fit(disp=False)
            models['sarimax'] = sarimax_results
        except:
            print("Warning: SARIMAX model failed to converge. Skipping.")

        # 4. LSTM (Deep Learning)
        print("Training LSTM model...")
        lstm_model = self.build_lstm_model(X_train_lstm.shape[1:])
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        lstm_model.fit(
            X_train_lstm, y_train_lstm,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        models['lstm'] = lstm_model

        self.models = models
        return models

    def build_lstm_model(self, input_shape):
        """Build and compile LSTM model."""
        model = keras.models.Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')
        return model

    def evaluate_models(self, datasets):
        """
        Evaluate the performance of all models.

        Parameters:
        datasets (dict): Dictionary of datasets for different models

        Returns:
        dict: Dictionary of evaluation metrics for each model
        """
        if not self.models:
            raise ValueError("Models not built. Call build_models() first.")

        # Unpack datasets
        X_train, X_test, y_train, y_test = datasets['standard']
        X_train_scaled, X_test_scaled, _, _ = datasets['scaled']
        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = datasets['lstm']

        results = {}
        self.model_predictions = {}

        # Evaluate each model
        for name, model in self.models.items():
            if name == 'random_forest' or name == 'gradient_boosting':
                y_pred = model.predict(X_test)

            elif name == 'sarimax':
                y_pred = model.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)

            elif name == 'lstm':
                y_pred = model.predict(X_test_lstm)

            else:
                continue

            # Store predictions
            self.model_predictions[name] = y_pred

            # Calculate metrics
            if name != 'lstm':
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
            else:
                mse = mean_squared_error(y_test_lstm, y_pred)
                mae = mean_absolute_error(y_test_lstm, y_pred)
                r2 = r2_score(y_test_lstm, y_pred)

            results[name] = {
                'MSE': mse,
                'RMSE': np.sqrt(mse),
                'MAE': mae,
                'R2': r2
            }

            print(f"{name} model metrics:")
            print(f"MSE: {mse:.4f}, RMSE: {np.sqrt(mse):.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

        # Calculate ensemble weights based on performance
        self.calculate_ensemble_weights(results)

        return results

    def calculate_ensemble_weights(self, results):
        """
        Calculate weights for ensemble model based on model performance.

        Parameters:
        results (dict): Dictionary of evaluation metrics for each model
        """
        # Use R2 score as the basis for weights
        weights = {}
        total_r2 = 0

        for name, metrics in results.items():
            # Only include models with positive R2
            if metrics['R2'] > 0:
                weights[name] = metrics['R2']
                total_r2 += metrics['R2']

        # Normalize weights
        if total_r2 > 0:
            for name in weights:
                weights[name] /= total_r2
        else:
            # If all models performed poorly, use equal weights
            for name in weights:
                weights[name] = 1.0 / len(weights)

        self.ensemble_weights = weights

    def create_ensemble_prediction(self, days=None):
        """
        Create ensemble prediction by combining all model predictions.

        Parameters:
        days (int): Number of days for the prediction, defaults to self.prediction_days

        Returns:
        pandas.DataFrame: DataFrame with ensemble predictions
        """
        if self.ensemble_weights is None:
            raise ValueError("Ensemble weights not calculated. Call evaluate_models() first.")

        if days is None:
            days = self.prediction_days

        # Generate future predictions for each model
        future_predictions = self.predict_future(days)

        # Create ensemble prediction
        ensemble_pred = pd.DataFrame()

        for day in range(days):
            weighted_sum = 0
            for name, weight in self.ensemble_weights.items():
                if name in future_predictions and day < len(future_predictions[name]):
                    weighted_sum += future_predictions[name][day] * weight

            date = self.data.index[-1] + timedelta(days=day + 1)
            ensemble_pred.loc[date, 'Prediction'] = weighted_sum

        return ensemble_pred

    def predict_future(self, days):
        """
        Generate future predictions using all models.

        Parameters:
        days (int): Number of days to predict into the future

        Returns:
        dict: Dictionary of predictions for each model
        """
        future_predictions = {}

        # Get the latest data for prediction
        latest_data = self.features.iloc[-1:].copy()

        for day in range(days):
            # For each model
            for name, model in self.models.items():
                if name not in future_predictions:
                    future_predictions[name] = []

                if name == 'random_forest' or name == 'gradient_boosting':
                    features = latest_data.drop(['Open', 'High', 'Low', 'Close'], axis=1)
                    pred = model.predict(features)[0]

                elif name == 'sarimax':
                    # For SARIMAX, predict one step ahead
                    if day == 0:
                        try:
                            pred = model.forecast(steps=days)
                        except:
                            print(f"Error occurred: {pred}")
                    else:
                        # Use the previous prediction
                        pred = future_predictions[name][-1]

                elif name == 'lstm':
                    # For LSTM, we need to prepare the sequence data
                    X_scaled = self.scaler.transform(
                        latest_data.drop(['Open', 'High', 'Low', 'Close'], axis=1))
                    # Reshape for LSTM input [samples, time steps, features]
                    # This is simplified and would need proper sequence preparation
                    X_seq = X_scaled.reshape(1, 1, X_scaled.shape[1])
                    pred = model.predict(X_seq)[0][0]

                else:
                    continue

                future_predictions[name].append(pred)

                # Update latest data for next day prediction
                # This is a simplified approach and would need more complex logic
                # for real-world accurate prediction
                new_row = latest_data.iloc[-1].copy()
                new_row['Close'] = pred
                # Update other features based on this prediction
                # ...

                # Append to latest data
                latest_data = pd.concat([latest_data, pd.DataFrame([new_row])])

        return future_predictions

    def run_full_pipeline(self, years=5, test_size=0.2):
        """
        Run the full prediction pipeline.

        Parameters:
        years (int): Number of years of historical data to fetch
        test_size (float): Proportion of data to use for testing

        Returns:
        pandas.DataFrame: DataFrame with ensemble predictions
        """
        print(f"Fetching data for {self.ticker}...")
        self.fetch_data(years=years)

        print("Creating features...")
        self.create_features()

        print("Preparing data for models...")
        datasets = self.prepare_data_for_models(test_size=test_size)

        print("Building models...")
        self.build_models(datasets)

        print("Evaluating models...")
        self.evaluate_models(datasets)

        print(f"Generating {self.prediction_days}-day predictions...")
        predictions = self.create_ensemble_prediction()

        return predictions

    def plot_predictions(self, predictions, historical_days=30):
        """
        Plot historical prices and future predictions.

        Parameters:
        predictions (pandas.DataFrame): DataFrame with predictions
        historical_days (int): Number of historical days to show
        """
        plt.figure(figsize=(12, 6))

        # Plot historical data
        historical = self.data[['Close']].iloc[-historical_days:]
        plt.plot(historical.index, historical['Close'], label='Historical Prices', color='blue')

        # Plot predictions
        plt.plot(predictions.index, predictions['Prediction'], label='Predicted Prices', color='red', linestyle='--')

        # Add confidence intervals (simplified)
        # In a real-world scenario, you would calculate proper confidence intervals
        upper_bound = predictions['Prediction'] * 1.05  # 5% higher
        lower_bound = predictions['Prediction'] * 0.95  # 5% lower
        plt.fill_between(predictions.index, lower_bound, upper_bound, color='red', alpha=0.2)

        plt.title(f'{self.ticker} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        return plt


# Example usage:
def predict_stock_price(ticker, prediction_days=30, years=5):
    """
    Predict future stock prices for a given ticker.

    Parameters:
    ticker (str): Stock ticker symbol
    prediction_days (int): Number of days to predict into the future
    years (int): Number of years of historical data to use

    Returns:
    pandas.DataFrame: DataFrame with predictions
    """
    predictor = StockPricePredictor(ticker, prediction_days=prediction_days)
    predictions = predictor.run_full_pipeline(years=years)

    print("\nPrediction Results:")
    print(predictions)

    # Plot results
    plt = predictor.plot_predictions(predictions)
    plt.show()

    return predictions


# Run prediction for a stock
if __name__ == "__main__":
    ticker = input("Enter stock ticker (e.g., AAPL): ")
    days = int(input("Enter number of days to predict: "))
    predictions = predict_stock_price(ticker, prediction_days=days)